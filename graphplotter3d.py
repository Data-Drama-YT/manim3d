from graph3d import Graph3D
import math

# High-level convenience API for common graph functions
class GraphPlotter:
    """High-level API for common mathematical functions"""
    
    def __init__(self, blender_graphs=None):
        """Initialize with Graph3D instance"""
        self.bg = blender_graphs if blender_graphs else Graph3D()
    
    def setup_cartesian_system(self, dimension=3, size=5, grid=True):
        """Setup a standard Cartesian coordinate system"""
        return self.bg.create_cartesian_system(
            dimension=dimension,
            size=size,
            grid=grid
        )
    
    def sine_wave(self, amplitude=1, frequency=1, phase=0, 
                 x_range=(-5, 5), color=(0, 0.8, 0.2, 1.0), with_text=True):
        """Plot a sine wave: y = A * sin(ω * x + φ)"""
        def func(x):
            return amplitude * math.sin(frequency * x + phase)
        
        equation = f"y = {amplitude} sin({frequency}x + {phase})"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def cosine_wave(self, amplitude=1, frequency=1, phase=0, 
                   x_range=(-5, 5), color=(0, 0.2, 0.8, 1.0), with_text=True):
        """Plot a cosine wave: y = A * cos(ω * x + φ)"""
        def func(x):
            return amplitude * math.cos(frequency * x + phase)
        
        equation = f"y = {amplitude} cos({frequency}x + {phase})"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def parabola(self, a=1, b=0, c=0, 
                x_range=(-5, 5), color=(0.8, 0.2, 0.0, 1.0), with_text=True):
        """Plot a parabola: y = a*x² + b*x + c"""
        def func(x):
            return a * x**2 + b * x + c
        
        equation = f"y = {a}x² + {b}x + {c}"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def line(self, slope=1, intercept=0, 
            x_range=(-5, 5), color=(0.5, 0.5, 0.5, 1.0), with_text=True):
        """Plot a straight line: y = mx + b"""
        def func(x):
            return slope * x + intercept
        
        equation = f"y = {slope}x + {intercept}"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def exponential(self, a=1, b=2, 
                   x_range=(-5, 5), color=(0.8, 0.5, 0.0, 1.0), with_text=True):
        """Plot an exponential function: y = a * b^x"""
        def func(x):
            return a * b**x
        
        equation = f"y = {a} · {b}^x"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def logarithm(self, base=10, scale=1, 
                 x_range=(0.1, 5), color=(0.0, 0.5, 0.8, 1.0), with_text=True):
        """Plot a logarithmic function: y = scale * log_base(x)"""
        def func(x):
            # Handle domain issues
            if x <= 0:
                return float('nan')
            return scale * math.log(x, base)
        
        equation = f"y = {scale} · log_{base}(x)"
        
        return self.bg.plot_function_2d(
            func=func,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def circle(self, radius=3, center=(0, 0, 0), 
              color=(0.8, 0.2, 0.8, 1.0), with_text=True):
        """Plot a circle using parametric equations"""
        def x_func(t):
            return center[0] + radius * math.cos(t)
        
        def y_func(t):
            return center[1] + radius * math.sin(t)
        
        def z_func(t):
            return center[2]
        
        equation = f"(x - {center[0]})² + (y - {center[1]})² = {radius}²"
        
        return self.bg.plot_parametric_3d(
            x_func=x_func,
            y_func=y_func,
            z_func=z_func,
            t_range=(0, 2*math.pi),
            color=color,
            equation_text=equation if with_text else None
        )
    
    def helix(self, radius=3, pitch=0.5, num_turns=3, 
             color=(0.2, 0.8, 0.6, 1.0), with_text=True):
        """Plot a 3D helix"""
        def x_func(t):
            return radius * math.cos(t)
        
        def y_func(t):
            return radius * math.sin(t)
        
        def z_func(t):
            return pitch * t / (2*math.pi)
        
        equation = "x = r·cos(t), y = r·sin(t), z = p·t/(2π)"
        
        return self.bg.plot_parametric_3d(
            x_func=x_func,
            y_func=y_func,
            z_func=z_func,
            t_range=(0, 2*math.pi*num_turns),
            samples=100*num_turns,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def paraboloid(self, a=0.25, b=0.25, 
                  x_range=(-5, 5), y_range=(-5, 5), 
                  color=(0.2, 0.7, 0.3, 1.0), with_text=True):
        """Plot a paraboloid: z = a*x² + b*y²"""
        def func(x, y):
            return a * x**2 + b * y**2
        
        equation = f"z = {a}x² + {b}y²"
        
        return self.bg.plot_function_3d(
            func=func,
            x_range=x_range,
            y_range=y_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def sinc_function(self, x_range=(-10, 10), y_range=(-10, 10), 
                     color=(0.1, 0.6, 0.9, 1.0), with_text=True):
        """Plot the 2D sinc function: z = sin(√(x² + y²)) / √(x² + y²)"""
        def func(x, y):
            r = math.sqrt(x**2 + y**2)
            if r < 0.001:
                return 1.0  # Limit as r approaches 0
            return math.sin(r) / r
        
        equation = "z = sin(√(x² + y²)) / √(x² + y²)"
        
        return self.bg.plot_function_3d(
            func=func,
            x_range=x_range,
            y_range=y_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def wave_interference(self, x_range=(-10, 10), y_range=(-10, 10), 
                         sources=2, frequency=1, 
                         color=(0.3, 0.3, 0.9, 1.0), with_text=True):
        """Plot wave interference pattern from point sources"""
        def func(x, y):
            result = 0
            # Create several point sources arranged in a circle
            for i in range(sources):
                angle = 2 * math.pi * i / sources
                source_x = 5 * math.cos(angle)
                source_y = 5 * math.sin(angle)
                
                # Distance from point to source
                r = math.sqrt((x - source_x)**2 + (y - source_y)**2)
                
                # Add wave contribution (decaying with distance)
                result += math.sin(frequency * r) / max(1, r**0.5)
            
            return result
        
        equation = f"{sources} point sources wave interference"
        
        return self.bg.plot_function_3d(
            func=func,
            x_range=x_range,
            y_range=y_range,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def sphere(self, radius=3, center=(0, 0, 0), 
              resolution=50, color=(0.1, 0.5, 0.9, 1.0), with_text=True):
        """Plot a sphere using implicit surface"""
        def func(x, y, z):
            return (x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2 - radius**2
        
        equation = f"(x - {center[0]})² + (y - {center[1]})² + (z - {center[2]})² = {radius}²"
        
        bounds = (
            center[0] - radius*1.2, center[0] + radius*1.2,
            center[1] - radius*1.2, center[1] + radius*1.2,
            center[2] - radius*1.2, center[2] + radius*1.2
        )
        
        return self.bg.plot_implicit_surface(
            func=func,
            bounds=bounds,
            resolution=resolution,
            color=color,
            equation_text=equation if with_text else None
        )
    
    def animate_sine_evolve(self, frames=100, start_frame=1, frequency_range=(0.5, 3),
                           x_range=(-5, 5), color=(0, 0.8, 0.2, 1.0), with_text=True):
        """Animate an evolving sine wave with changing frequency"""
        def func(x, progress):
            # Frequency evolves from min to max
            current_freq = frequency_range[0] + progress * (frequency_range[1] - frequency_range[0])
            return math.sin(current_freq * x)
        
        equation = "y = sin(ωx), ω evolving"
        
        return self.bg.animate_function_2d(
            func=func,
            frames=frames,
            start_frame=start_frame,
            x_range=x_range,
            color=color,
            equation_text=equation if with_text else None,
            animation_type="evolve"
        )
    
    # 3D Graph Plotting Systems

    def animate_3d_wave(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.2, 0.6, 0, 1.0), with_text=True):
        """Animate a 3D wave propagation"""
        def func(x, y, progress=1):
            time = progress * 5  # Scale time for animation effect
            r = math.sqrt(x**2 + y**2)
            return 2 * math.sin(r - time) / max(1, r**0.7)
        
        equation = "z = sin(√(x² + y²) - t)"
        
        return self.bg.animate_function_3d(func)
    
    def animate_3d_paraboloid(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.1, 0.5, 1.0, 1.0), with_text=True):
        """Animate a 3D Paraboloid growing"""
        def func(x, y, progress=1):
            return progress * (x**2 + y**2) * 0.1  # Scaled to stay visible

        equation = "z = x² + y²"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_ripple(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.8, 0.4, 1.0, 1.0), with_text=True):
        """Animate 3D ripples propagating outward"""
        def func(x, y, progress=1):
            r = math.sqrt(x**2 + y**2)
            return math.sin(5*r - progress*10) * 0.5

        equation = "z = sin(5√(x²+y²) - t)"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    def animate_3d_saddle(self, frames=100, start_frame=1, 
                       x_range=(-3, 3), y_range=(-3, 3), 
                       color=(1.0, 0.5, 0.2, 1.0), with_text=True):
        """Animate 3D saddle shape"""
        def func(x, y, progress=1):
            return progress * (x**2 - y**2) * 0.2

        equation = "z = x² - y²"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_helicoid(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.5, 0.9, 0.2, 1.0), with_text=True):
        """Animate a 3D helicoid surface"""
        def func(x, y, progress=1):
            return progress * math.atan2(y, x)

        equation = "z = atan2(y, x)"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_gaussian_bump(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(1.0, 0.8, 0.2, 1.0), with_text=True):
        """Animate a 3D Gaussian bump"""
        def func(x, y, progress=1):
            return progress * math.exp(-(x**2 + y**2))

        equation = "z = e^(-(x² + y²))"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)

    def animate_3d_wave_interference(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.2, 0.2, 1.0, 1.0), with_text=True):
        """Animate two interfering waves"""
        def func(x, y, progress=1):
            return math.sin(x + progress*5) + math.sin(y + progress*5)

        equation = "z = sin(x + t) + sin(y + t)"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_volcano(self, frames=100, start_frame=1, 
                       x_range=(-3, 3), y_range=(-3, 3), 
                       color=(0.8, 0.3, 0.1, 1.0), with_text=True):
        """Animate a 3D volcano eruption"""
        def func(x, y, progress=1):
            r = math.sqrt(x**2 + y**2)
            return (1 - r) * 4 * progress if r < 1 else 0

        equation = "z = (1 - √(x²+y²)) * eruption"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    def animate_3d_breathing_sphere(self, frames=100, start_frame=1, 
                       x_range=(-1.5, 1.5), y_range=(-1.5, 1.5), 
                       color=(0.1, 0.6, 1.0, 1.0), with_text=True):
        """Animate a sphere that breathes"""
        def func(x, y, progress=1):
            if x**2 + y**2 > 1:
                return 0  # Hide outside sphere
            return math.sqrt(1 - x**2 - y**2) * (0.8 + 0.2 * math.sin(progress * 2 * math.pi))

        equation = "z = √(1-x²-y²) * breathing factor"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_morphing_torus(self, frames=100, start_frame=1, 
                       x_range=(-2, 2), y_range=(-2, 2), 
                       color=(0.7, 0.4, 1.0, 1.0), with_text=True):
        """Animate a torus (donut shape) that morphs"""
        def func(x, y, progress=1):
            r = math.sqrt(x**2 + y**2)
            return (0.5 - (r - 1)**2) * (1 + 0.3 * math.sin(progress * 2 * math.pi))

        equation = "z = (0.5 - (√(x²+y²) - 1)²) * morphing"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    def animate_3d_heart(self, frames=100, start_frame=1, 
                       x_range=(-2, 2), y_range=(-2, 2), 
                       color=(1.0, 0.2, 0.4, 1.0), with_text=True):
        """Animate a 3D pulsating heart"""
        def func(x, y, progress=1):
            z = (x**2 + y**2 - 1)**3 - x**2 * y**3
            # If abs(z) is less than 1, calculate the value, else set z to a default value like 0.
            if abs(z) < 1:
                return 0.5 * math.sin(progress * 2 * math.pi) * (1 - abs(z))
            else:
                return 0  # Fallback value instead of None

        equation = "Heart equation, pulsating with time"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                            x_range=x_range, y_range=y_range, color=color,
                                            equation_text=equation if with_text else None)

    
    def animate_3d_black_hole(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(0.0, 0.0, 0.0, 1.0), with_text=True):
        """Animate a black hole gravity distortion"""
        def func(x, y, progress=1):
            r = math.sqrt(x**2 + y**2)
            if r < 0.1:
                return 0  # Singular point (too sharp)
            distortion = 1 / (r**1.5 + 0.2)
            return distortion * (1 + 0.2 * math.sin(progress * 2 * math.pi))

        equation = "z = gravitational distortion field"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_spiral_galaxy(self, frames=150, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(1.0, 0.8, 0.3, 1.0), with_text=True):
        """Animate a rotating spiral galaxy"""
        def func(x, y, progress=1):
            angle = math.atan2(y, x)
            radius = math.sqrt(x**2 + y**2)
            spiral = math.sin(5 * angle + radius - progress * 2 * math.pi)
            return spiral * (1 / (1 + radius))

        equation = "z = sin(5θ + r - time) / (1 + r)"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_mandelbrot_slice(self, frames=100, start_frame=1, 
                       x_range=(-2, 2), y_range=(-2, 2), 
                       color=(0.4, 0.2, 0.6, 1.0), with_text=True):
        """Animate a morphing Mandelbrot set slice"""
        def func(x, y, progress=1):
            c = complex(x, y)
            z = 0
            count = 0
            max_iter = int(20 + 80 * progress)
            while abs(z) <= 2 and count < max_iter:
                z = z*z + c
                count += 1
            return count / max_iter

        equation = "z = Mandelbrot iteration depth"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_shockwave(self, frames=100, start_frame=1, 
                       x_range=(-5, 5), y_range=(-5, 5), 
                       color=(1.0, 0.5, 0.0, 1.0), with_text=True):
        """Animate a shockwave expanding outward"""
        def func(x, y, progress=1):
            r = math.sqrt(x**2 + y**2)
            wavefront = math.sin(10 * (r - progress * 5))
            envelope = math.exp(-((r - progress * 5)**2) * 5)
            return wavefront * envelope

        equation = "z = sin(10(r - t)) * exp(-5(r - t)²)"

        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    def animate_3d_tornado_vortex(self, frames=100, start_frame=1, 
                               x_range=(-5, 5), y_range=(-5, 5), 
                               color=(0.2, 0.6, 1, 1.0), with_text=True):
        """Animate a 3D tornado vortex"""
        def func(x, y, progress=1):
            time = progress * 10  # Scale the time for animation
            r = math.sqrt(x**2 + y**2)
            z = math.sin(r - time) * math.exp(-r / 5.0)  # Spiral down effect
            return z
        
        equation = "z = sin(√(x² + y²) - t) * exp(-√(x² + y²)/5)"
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_northern_lights(self, frames=100, start_frame=1, 
                                x_range=(-5, 5), y_range=(-5, 5), 
                                color=(0.2, 0.8, 0.5, 1.0), with_text=True):
        """Animate a 3D northern lights effect"""
        def func(x, y, progress=1):
            time = progress * 5  # Animate over time
            z = math.sin(x * 2 + time) * math.cos(y * 2 + time) * math.exp(-math.sqrt(x**2 + y**2) / 5)
            return z
        
        equation = "z = sin(2x + t) * cos(2y + t) * exp(-√(x² + y²)/5)"
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_expanding_bubble(self, frames=100, start_frame=1, 
                                 x_range=(-5, 5), y_range=(-5, 5), 
                                 color=(0.8, 0.2, 0.8, 1.0), with_text=True):
        """Animate an expanding bubble collapsing in 3D"""
        def func(x, y, progress=1):
            r = math.sqrt(x**2 + y**2)
            z = math.exp(-r * (1 - progress))  # Bubble expanding and collapsing effect
            return z
        
        equation = "z = exp(-√(x² + y²) * (1 - t))"
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_ocean_waves(self, frames=100, start_frame=1, 
                            x_range=(-5, 5), y_range=(-5, 5), 
                            color=(0.2, 0.6, 1, 0.8), with_text=True):
        """Animate ocean waves rolling in 3D"""
        def func(x, y, progress=1):
            time = progress * 2  # Scale time for wave movement
            r = math.sqrt(x**2 + y**2)
            z = math.sin(r - time) * math.exp(-r / 5.0)  # Rolling wave effect
            return z
        
        equation = "z = sin(√(x² + y²) - t) * exp(-√(x² + y²)/5)"
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)
    
    def animate_3d_linear_regression(self, frames=100, start_frame=1,
                              x_range=(-3, 3), y_range=(-3, 3),
                              color=(0.3, 0.7, 0.9, 0.8), with_text=True):
        """Animate a linear regression plane fitting to noisy 3D data"""
        def func(x, y, progress=1):
            # Base linear relationship
            true_value = 0.5*x + 1.2*y
            # Noise that reduces as regression converges
            noise = (1 - progress) * math.sin(x*10)*math.cos(y*10)*0.5
            return true_value + noise
        
        equation = "z = 0.5x + 1.2y + ε(1-t)\nLinear regression converging"
        
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)

    def animate_3d_polynomial_regression(self, frames=100, start_frame=1,
                                    x_range=(-3, 3), y_range=(-3, 3),
                                    color=(0.9, 0.4, 0.2, 0.8), with_text=True):
        """Animate polynomial regression fitting with increasing degrees"""
        def func(x, y, progress=1):
            # Animate degree increasing from 1 to 4
            degree = int(1 + progress * 3)
            return sum((0.5**n) * (x**n + y**n) for n in range(1, degree+1))
        
        equation = "z = Σ (0.5ⁿ)(xⁿ + yⁿ) for n=1→4\nPolynomial regression"
        
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)

    def animate_3d_regularization(self, frames=100, start_frame=1,
                                x_range=(-2, 2), y_range=(-2, 2),
                                color=(0.7, 0.2, 0.5, 0.8), with_text=True):
        """Animate regularization effects on regression surface"""
        def func(x, y, progress=1):
            # Base quadratic surface
            base = x**2 + x*y + y**2
            # Regularization term that grows over time
            reg = progress * 5 * (abs(x) + abs(y))  # L1 regularization
            return base + reg
        
        equation = "z = x² + xy + y² + λ|θ|\nL1 regularization increasing"
        
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)

    def animate_3d_logistic_regression(self, frames=100, start_frame=1,
                                    x_range=(-3, 3), y_range=(-3, 3),
                                    color=(0.2, 0.8, 0.4, 0.8), with_text=True):
        """Animate logistic regression decision boundary sharpening"""
        def func(x, y, progress=1):
            # Decision boundary becomes sharper as training progresses
            sharpness = 2 + progress * 8  # Increasing slope
            return 1 / (1 + math.exp(-sharpness*(x + 0.5*y - 1)))
        
        equation = "z = σ(k(t)(x + 0.5y - 1))\nDecision boundary sharpening"
        
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)

    def animate_3d_residuals(self, frames=100, start_frame=1,
                        x_range=(-3, 3), y_range=(-3, 3),
                        color=(0.5, 0.5, 0.1, 0.8), with_text=True):
        """Animate residual errors during regression fitting"""
        def func(x, y, progress=1):
            true_value = 0.8*x - 0.5*y
            # Residuals that decrease over time
            residual = (1-progress) * math.sin(x*10)*math.cos(y*10)
            return true_value + residual
        
        equation = "z = 0.8x - 0.5y + (1-t)·noise\nResiduals decreasing"
        
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)

    def animate_3d_overfitting(self, frames=100, start_frame=1,
                            x_range=(-3, 3), y_range=(-3, 3),
                            color=(0.8, 0.3, 0.6, 0.8), with_text=True):
        """Animate the progression from underfitting to overfitting"""
        def func(x, y, progress=1):
            # True underlying function
            true_func = math.sin(x) + math.cos(y)
            # Start with underfit, progress to overfit
            if progress < 0.3:  # Underfit phase
                return 0.5*x + 0.3*y
            elif progress < 0.6:  # Good fit
                return true_func
            else:  # Overfit phase
                return true_func + 0.5*math.sin(5*x)*math.cos(5*y)*(progress-0.6)*5
        
        equation = "z: Underfit → Good fit → Overfit\nModel complexity increasing"
        
        return self.bg.animate_function_3d(func, frames=frames, start_frame=start_frame,
                                        x_range=x_range, y_range=y_range, color=color,
                                        equation_text=equation if with_text else None)














