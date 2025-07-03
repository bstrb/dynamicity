import tkinter as tk
import math

class EwaldSphereApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Ewald Sphere and Real-Space Visualization")

        # Reciprocal space canvas
        self.canvas = tk.Canvas(self.root, width=800, height=600, bg="white")
        self.canvas.grid(row=0, column=0, columnspan=3)

        # Real-space canvas
        self.real_space_canvas = tk.Canvas(self.root, width=400, height=400, bg="lightgray")
        self.real_space_canvas.grid(row=0, column=3)

        # HKL panel
        self.hkl_text = tk.Text(self.root, width=30, height=20)
        self.hkl_text.grid(row=0, column=4)
        self.hkl_text.insert(tk.END, "HKL Spots Satisfying Bragg's Law:\n")

        # Controls
        self.create_controls()

        # Default parameters
        self.wavelength = 1.0  # Angstrom
        self.lattice_spacing = 2.0  # Angstrom
        self.rotation = 0  # Degrees

        # Initial visualization
        self.update_visualization()

    def create_controls(self):
        # Wavelength slider
        tk.Label(self.root, text="X-ray Wavelength (Å)").grid(row=1, column=0)
        self.wavelength_slider = tk.Scale(self.root, from_=0.5, to=2.0, resolution=0.01, orient=tk.HORIZONTAL, command=self.update_wavelength)
        self.wavelength_slider.set(1.0)
        self.wavelength_slider.grid(row=2, column=0)

        # Lattice spacing slider
        tk.Label(self.root, text="Lattice Spacing (Å)").grid(row=1, column=1)
        self.lattice_spacing_slider = tk.Scale(self.root, from_=1.0, to=5.0, resolution=0.1, orient=tk.HORIZONTAL, command=self.update_lattice_spacing)
        self.lattice_spacing_slider.set(2.0)
        self.lattice_spacing_slider.grid(row=2, column=1)

        # Rotation slider
        tk.Label(self.root, text="Lattice Rotation (°)").grid(row=1, column=2)
        self.rotation_slider = tk.Scale(self.root, from_=0, to=360, resolution=1, orient=tk.HORIZONTAL, command=self.update_rotation)
        self.rotation_slider.set(0)
        self.rotation_slider.grid(row=2, column=2)

    def update_wavelength(self, value):
        self.wavelength = float(value)
        self.update_visualization()

    def update_lattice_spacing(self, value):
        self.lattice_spacing = float(value)
        self.update_visualization()

    def update_rotation(self, value):
        self.rotation = float(value)
        self.update_visualization()

    def update_visualization(self):
        # Update reciprocal space visualization
        self.canvas.delete("all")
        base_radius = 200  # Base radius for a wavelength of 1.0 Å
        sphere_radius = base_radius / self.wavelength
        center_x, center_y = 400, 300
        self.canvas.create_oval(center_x - sphere_radius, center_y - sphere_radius,
                                center_x + sphere_radius, center_y + sphere_radius,
                                outline="blue", width=2)
        self.canvas.create_line(center_x, center_y, center_x + sphere_radius, center_y,
                                arrow=tk.LAST, fill="green", width=2)
        self.draw_reciprocal_lattice(center_x, center_y, sphere_radius)

        # Update real-space visualization
        self.update_real_space()

    def draw_reciprocal_lattice(self, center_x, center_y, sphere_radius):
        points = []
        max_points = 10
        lattice_scale = 200 / self.lattice_spacing  # Adjust lattice scale based on lattice spacing

        self.hkl_text.delete(1.0, tk.END)  # Clear HKL panel
        self.hkl_text.insert(tk.END, "HKL Spots Satisfying Bragg's Law:\n")

        for i in range(-max_points, max_points + 1):
            for j in range(-max_points, max_points + 1):
                x = i * lattice_scale
                y = j * lattice_scale
                theta = math.radians(self.rotation)
                x_rot = x * math.cos(theta) - y * math.sin(theta)
                y_rot = x * math.sin(theta) + y * math.cos(theta)
                canvas_x = center_x + x_rot
                canvas_y = center_y - y_rot
                points.append((canvas_x, canvas_y, i, j))

        for x, y, h, k in points:
            dist_to_center = math.sqrt((x - center_x)**2 + (y - center_y)**2)
            if abs(dist_to_center - sphere_radius) < 5:
                color = "red"
                size = 6
                self.hkl_text.insert(tk.END, f"h={h}, k={k}, \u03b8={self.calculate_bragg_theta(h, k)}\n")

                # Draw Bragg angle on reciprocal space
                theta = math.radians(self.calculate_bragg_theta(h, k))
                bragg_x = center_x + sphere_radius * math.cos(theta)
                bragg_y = center_y - sphere_radius * math.sin(theta)
                self.canvas.create_line(center_x, center_y, bragg_x, bragg_y, fill="red", width=2, dash=(4, 2))
                self.canvas.create_text((bragg_x + center_x) / 2, (bragg_y + center_y) / 2, text=f"\u03b8={self.calculate_bragg_theta(h, k):.1f}°", fill="black")
            else:
                color = "black"
                size = 3
            self.canvas.create_oval(x - size, y - size, x + size, y + size, fill=color)

    def calculate_bragg_theta(self, h, k):
        d_spacing = self.lattice_spacing
        reciprocal_distance = math.sqrt(h**2 + k**2) / d_spacing
        try:
            theta = math.asin(self.wavelength * reciprocal_distance / 2)  # Bragg's Law
            return math.degrees(theta)  # Convert to degrees
        except ValueError:
            return "Invalid"

    def update_real_space(self):
        self.real_space_canvas.delete("all")
        self.real_space_canvas.create_line(200, 0, 200, 200, arrow=tk.LAST, fill="blue", width=2)
        lattice_spacing = self.lattice_spacing * 40  # Scale the lattice spacing

        for i in range(5):
            for j in range(5):
                x = 200 + (i - 2) * lattice_spacing
                y = 200 + (j - 2) * lattice_spacing
                theta = math.radians(self.rotation)
                x_rot = (x - 200) * math.cos(theta) - (y - 200) * math.sin(theta) + 200
                y_rot = (x - 200) * math.sin(theta) + (y - 200) * math.cos(theta) + 200
                self.real_space_canvas.create_oval(x_rot - 5, y_rot - 5, x_rot + 5, y_rot + 5, fill="black")

if __name__ == "__main__":
    root = tk.Tk()
    app = EwaldSphereApp(root)
    root.mainloop()
