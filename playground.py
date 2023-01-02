from typing import Callable
from math import ceil, sqrt, pi, sin, cos, asin
from random import Random
from abc import ABC, abstractmethod
from typing import Iterator, Tuple
from statistics import NormalDist
from os import makedirs
from os.path import join, exists
from dataclasses import dataclass, field
from subprocess import run, STDOUT, PIPE


twopi = 2*pi

Boundaries = Tuple[float, float, float]
Point = Tuple[float, float, float]
RGB = Tuple[int, int, int]
Color = Tuple[float, float, float]

StatePDF = Callable[[Point], float]
ModelPDF = Callable[[Point, float], float] # point, time -> pdf

def state_at_moment(model: ModelPDF, t: float) -> StatePDF:
    return lambda p: model(p, t)

class PointSower(ABC):
    @abstractmethod
    def next_point(boundaries: Boundaries) -> Point: ...

SowerFactory = Callable[[StatePDF, float], PointSower]

class NaiveMonteCarloSower(PointSower):
    '''
    Chooses point uniformly from boundaries, checks its density, picks uniformly from [0, 1), if pick <= prob -> return, else -> repeat
    '''
    
    @abstractmethod
    def next_point(boundaries: Boundaries) -> Point: 
        self._random = val

    def __init__(self, model: StatePDF, random: Random=None):
        self.model = model
        self.random = random or Random()
    
    def next_point(self, boundaries: Boundaries):
        def pick(bound_comp: float):
            return self.random.uniform(-1*bound_comp, bound_comp)
        while True:
            (x, y, z) = tuple(map(pick, boundaries))
            prob = self.model((x, y, z))
            toss = self.random.uniform(0, 1)
            if toss <= prob:
                return (x, y, z)

class Playground:
    def __init__(self, boundaries: Boundaries or float, granularity: float=0.5, threshold: float=0.8, corpuscule_count: int=-1):
        try:
            x, y, z = boundaries
        except TypeError:
            boundaries = float(boundaries)
            boundaries = (boundaries, boundaries, boundaries)
        self.boundaries = tuple(map(abs, boundaries))
        '''
        Playground is a box (including borders and walls) between (-x, -y, -z) and (x, y, z)=boundaries; 
        if given as single number c, it is treated as (c, c, c).
        '''
        granularity=float(granularity)
        self.granularity=granularity
        '''
        Radius of single sphere
        '''
        self.threshold = abs(float(threshold))
        '''
        Blob threshold
        '''
        if corpuscule_count <= 500:
            print("Requested", corpuscule_count, "corpuscules")
            #per_x, per_y, per_z = tuple(map(lambda a: ceil(max(1.0, a/(0.85*granularity))), boundaries))
            #print("Per dimension:", (per_x, per_y, per_z))
            #corpuscule_count = per_x*per_y*per_z
            boundaries_volume = 8*self.boundaries[0]*self.boundaries[1]*self.boundaries[2]
            sphere_volume = 4*pi*granularity*granularity*granularity/3
            print("Boundaries ratio:", boundaries_volume/sphere_volume)
            corpuscule_count = max(1000, ceil(boundaries_volume/sphere_volume))
            print("New count:", corpuscule_count)
        self.corpuscule_count = corpuscule_count
        '''
        Numer of blob components used to render the playground; if negative, will try to figure out sensible number for given boundaries
        '''
    
    def sow(self, generator: PointSower) -> Iterator[Point]:
        return ( generator.next_point(self.boundaries) for i in range(self.corpuscule_count) )

Pigmenter = Callable[[int, int, Point], str] #returns pigment expression

def scale(s: float, rgb: RGB) -> Color:
    return tuple(map(lambda x: float(s)*x/(255), rgb))

def fading(base_color: RGB, transparency=0.4):
    def pigmenter(i, total, point):
        position = (1.0+i)/total
        r, g, b = scale(position, base_color)
        return "pigment { color <"+str(r)+", "+str(g)+", "+str(b)+"> transmit "+str(transparency)+" }"
    return pigmenter

def height(base_color, y_limit, transparency=0.4):
    def pigmenter(i, total, point):
        position = min(1.0, max(0.0, (point[1] + y_limit)/(2*y_limit))) # it is actually (y-(-(limit))/2limit)
        r, g, b = scale(position, base_color)
        return "pigment { color <"+str(r)+", "+str(g)+", "+str(b)+"> transmit "+str(transparency)+" }"
    return pigmenter

class SceneRenderer:
    def __init__(self, playground: Playground, points: [Point], 
                        pic_dim=(1024,780), 
                            #camera = (x, y, z) = all-positive corner of boundaries -> x*=a, y*=b -> (x, y, z)*c
                            #light=d*camera
                            #params=(a, b, c, d)
                            #colors=(light, background, corpuscule)
                        params = (0.5, 1.2, 1.5, 1.2), 
                        colors=("White", "Gray50"),
                        pigmenter: Pigmenter=None):
        self.playground = playground
        self.points = list(points)
        self.dimensions = pic_dim
        self.params = params
        self.colors = colors
        if pigmenter is None:
            pigmenter = fading((0, 0, 255))
        self.pigmenter = pigmenter
        
    def scene(self):
        (a, b, c, d, *whatever) = self.params
        (light, background, *whatever) = self.colors
        cx, cy, cz = self.playground.boundaries
        cx *= a*c
        cy *= b*c
        cz *= c
        lx, ly, lz = d*cx, d*cy, d*cz
        return """
#include "colors.inc"
camera {
    location <"""+str(cx)+", "+str(cy)+", "+str(cz)+""">
    look_at 0
}
background { color """+background+""" }
light_source { <"""+str(lx)+", "+str(ly)+", "+str(lz)+"> color "+light+""" }
"""+self.blob()

    def blob(self):
        (*whatever, corpuscule) = self.params
        r = self.playground.granularity
        total = len(self.points)
        spheres = "\n    ".join(list(
            "sphere { <"+str(x)+", "+str(y)+", "+str(z)+">, "+str(r)+", 1 "+self.pigmenter(i, total, (x, y, z))+" }"
            for i, (x, y, z) in enumerate(self.points)
        ))
        return """blob {
    threshold """+str(
            self
                .playground
                .threshold
            )+" \n    " + spheres+"""
    finish {phong 1}
}
    """

def l(*coord):
    return sqrt(sum(map(lambda a: a*a, coord)))

def ripple_model(peak_height, peak_radius, wave_width, variation: float=0.5):
    '''
    Ripple, centered at 0, 0, on XZ plane.
    peak_height - largest y coordinate of any corpuscule on the wave
    peak_radius - distance from current position (x, z) of wave peak to (0,0,0)
    wave_width - width of half of wave (equivalent of pi/2 of sinus) in playground units
    replacement  - if sinus-based prob is too small, return this; effectively, probability of fiding corpuscule in vacuum
    '''
    def prob_dens(p):
        xy_l = l(p[0], p[2])
        dist_from_peak = abs(peak_radius - xy_l)
        in_sin_units = pi*dist_from_peak / (2*wave_width)
        expected_height = sin(in_sin_units)
        # we should take expected=(x, expected height, z), then calc abs(l(p-expected)), then return value of gauss(0, wave_width)(abs(...))
        #p-expected=(0, y-exp, 0), so l(p-exp) = (y-exp)^2
        #so we simplify it a bit
        y_diff = p[1] - expected_height # todo maybe scale it by expectdd height?
        from_prob_peak = y_diff*y_diff
        return NormalDist(0, variation).pdf(from_prob_peak)
    return prob_dens

SeedGenerator = Callable[[int], int] # previous_seed -> next_seed

increment = lambda x: x+1

@dataclass
class AnimationSetup:
    fps: int=15
    duration: float=3.0
    start_time: float=0.0
    start_seed: int=0xDEADBEEF
    seed_generator: SeedGenerator=increment
    wh: (int, int)=(640, 480)
    quality: int=5
    display: bool=False
    renderer_kwargs: dict=field(default_factory=dict)
    
    @property
    def cli(self):
        return ("" if self.display else "-Display ")+"-W"+str(self.wh[0])+" -H"+str(self.wh[1])+" +Q"+str(self.quality)

class Animation:
    def __init__(self, workspace: str, playground: Playground, sower_factory: SowerFactory, model: ModelPDF, setup: AnimationSetup=None):
        self.workspace = workspace
        self.playground = playground
        self.sower_factory = sower_factory
        self.model = model
        self.setup = setup or AnimationSetup()
        
        makedirs(join(workspace, "source"), exist_ok=True)
        makedirs(join(workspace, "rendered"), exist_ok=True)
        makedirs(join(workspace, "logs"), exist_ok=True)
    
    @property
    def setup(self) -> AnimationSetup:
        return self._setup
    
    @setup.setter
    def setup(self, val: AnimationSetup):
        assert(val is not None)
        self._setup = val
        print("Given setup:", val)
        self.frame_duration = 1.0/val.fps
        self.frame_count = ceil(val.duration/self.frame_duration)
        print("There will be", self.frame_count, "frames, each lasting", self.frame_duration, "s")
    
    def animate(self, run_rendering: bool=None):
        print("=== ANIMATION START === ")
        print("Workspace:", self.workspace)
        print("Playground:", self.playground)
        print("Setup:", self.setup)
        max_str = str(self.frame_count)
        max_no_digits = len(max_str)
        current_seed = None
        render_script_path = join(self.workspace, "render.sh")
        
        with open(render_script_path, "w") as render_script:
            render_script.write("set -ex\n\n")
            for frame_idx in range(self.frame_count):
                idx_str = str(frame_idx).zfill(max_no_digits)
                print("Frame #"+idx_str+"/"+max_str)
                basename = "frame_"+idx_str+"_"+max_str
                current_seed = self.setup.start_seed if current_seed is None else self.setup.seed_generator(current_seed)
                current_time = self.setup.start_time + frame_idx*self.frame_duration
                state_model = state_at_moment(self.model, current_time)
                sower = self.sower_factory(state_model, Random(current_seed))
                pov_path = join(join(self.workspace, "source"), basename+".pov")
                renders_path = join(self.workspace, "rendered")
                png_path = join(renders_path, basename+".png")
                logs_path = join(join(self.workspace, "logs"), basename+".txt")
                animation_path = join(self.workspace, "animation.gif")
                
                skip_frame = exists(pov_path)
                
                if skip_frame:
                    print("Skipping source preparation")
                else:
                    print("Sowing points")
                    points = self.playground.sow(sower)
                    print("Preparing POV-Ray sources")
                    renderer = SceneRenderer(self.playground, points, **self.setup.renderer_kwargs)
                    with open(pov_path, "w") as pov_source:
                        pov_source.write(renderer.scene())
                
                skip_render = exists(png_path) and skip_frame # if sources changed, render even if picture exists
                
                if skip_render:
                    print("Frame has already been rendered, commenting this line out")
                    render_script.write("#")
                print("Adding povray command to render script")
                render_script.write("povray "+pov_path+" -O"+png_path+" "+self.setup.cli+" 2> "+logs_path+"\n")
                print("Done")
            print("Adding GIF generator")
            #https://github.com/ImageMagick/ImageMagick/issues/396#issuecomment-319569255
            render_script.write("convert -delay "+str(ceil(self.frame_duration/0.01))+" -alpha deactivate -loop 0 "+renders_path+"/*.png "+animation_path)
        run("chmod +x "+render_script_path, shell=True, check=True, stdout=PIPE, stderr=STDOUT)
        if run_rendering is None:
            run_rendering = exists(animation_path)
            print("GIF exists?", run_rendering)
        if run_rendering:
            print("Running the rendering script")
            
            #cmd = "/usr/bin/bash "+render_script_path
            cmd = render_script_path
            print("$", cmd)
            run(cmd, shell=True, check=True, stdout=PIPE, stderr=STDOUT)
        print("=== ANIMATION END   ===")
        print()
        print()
        print()


def spiral_model(strands: int, 
            r_over_h_t: Callable[[float, float], float], 
            phase_over_h: Callable[[float], float], 
            phase_over_t: Callable[[float], float], 
            width_over_t: Callable[[float], float],
            start_phase: float=0):
    assert(strands>0)
    assert(phase_over_h is not None)
    assert(phase_over_t is not None)
    is_phase = lambda a: a >=0 and a < twopi
    def to_phase(angle, period):
        return angle - int(1.0*angle/period)*period
    assert(is_phase(start_phase))
    angle_between_strands = twopi / strands
    def pdf(p: Point, t: float):
        point_r = sqrt(p[0]*p[0] + p[2]*p[2]) + 0.0000001
        angle = asin(p[2]/point_r)
        point_h = p[1]
        point_phase = to_phase(angle, angle_between_strands)
        expected_phase = to_phase(start_phase + phase_over_h(point_h) + phase_over_t(t), angle_between_strands)
        expected_r = r_over_h_t(point_h, t)
        p2d = (p[0], p[2])
        e2d = (expected_r*cos(expected_phase), expected_r*sin(expected_phase))
        diff = (p2d[0] - e2d[0], p2d[1]-e2d[1])
        distance = sqrt(diff[0]*diff[0] + diff[1]*diff[1])
        return NormalDist(0, width_over_t(t)).pdf(distance)
    return pdf

if __name__ == "__main__":
    [
        a.animate(run_rendering=True)
        for a in (
            Animation(
                "./ripple/a",
                Playground((20.0, 3.0, 20.0), granularity=0.35),
                lambda model, random: NaiveMonteCarloSower(model, random),
                lambda p, t: ripple_model(peak_height=4, peak_radius=(0.15+1.5*t), wave_width=4)(p),
                AnimationSetup(
                    renderer_kwargs = {
                        "params": (1.5, 7.5, 1.5, 1.2),
                        "pigmenter": height([255, 0, 255], 3.0)
                    }
                )
            ),
            Animation(
                "./ripple/b",
                Playground((20.0, 3.0, 20.0), granularity=0.15),
                lambda model, random: NaiveMonteCarloSower(model, random),
                lambda p, t: ripple_model(peak_height=4, peak_radius=(0.15+0.75*t), wave_width=1)(p),
                AnimationSetup(
                    renderer_kwargs = {
                        "params": (1.5, 7.5, 1.5, 1.2),
                        "pigmenter": height([255, 0, 255], 3.0)
                    }
                )
            ),
            Animation(
                "./ripple/c",
                Playground((30.0, 8.0, 30.0), granularity=0.35),
                lambda model, random: NaiveMonteCarloSower(model, random),
                lambda p, t: ripple_model(peak_height=6, peak_radius=(0.25*t), wave_width=2)(p),
                AnimationSetup(
                    fps=60,
                    duration=20.0,
                    wh=(1024,720),
                    renderer_kwargs = {
                        "params": (1.5, 7.5, 1.5, 1.2),
                        "pigmenter": height([0, 0, 255], 2.0)
                    }
                )
            ),
            Animation(
                "./spiral/a",
                Playground((5.0, 20.0, 5.0), granularity=0.35),
                lambda model, random: NaiveMonteCarloSower(model, random),
                spiral_model(
                    strands=3,
                    r_over_h_t=lambda h, t: 3.0,
                    phase_over_h=lambda h: h*twopi/10.0, # 2hpi/x is "one full turn every x units"
                    phase_over_t=lambda h: h*twopi/2.5,
                    width_over_t=lambda t: 1.2,
                    start_phase=0.0
                ),
                AnimationSetup(
                    fps=15,
                    duration=5.0,
                    wh=(640,480),
                    renderer_kwargs = {
                        "params": (0.9, 1.2, 1.5, 1.2),
                        "pigmenter": height([196, 127, 64], 20.0)
                    }
                )
            ),
            Animation(
                "./spiral/b",
                Playground((20.0, 40.0, 20.0), granularity=0.8),
                lambda model, random: NaiveMonteCarloSower(model, random),
                spiral_model(
                    strands=3,
                    r_over_h_t=lambda h, t: 10.0,
                    phase_over_h=lambda h: h*twopi/23.4, # 2hpi/x is "one full turn every x units"
                    phase_over_t=lambda h: h*twopi/2.5,
                    width_over_t=lambda t: 1.0,
                    start_phase=0.0
                ),
                AnimationSetup(
                    fps=15,
                    duration=5.0,
                    wh=(640,480),
                    renderer_kwargs = {
                        "params": (0.9, 1.0, 1.5, 1.2),
                        "pigmenter": height([196, 127, 64], 20.0)
                    }
                )
            )
        )
    ]
