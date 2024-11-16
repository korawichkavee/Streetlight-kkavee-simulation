import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Step 1: Load and parse the IES file
def load_ies(file_path):
    """Parse IES file and return photometric data."""
    return parse_ies(file_path)

def parse_ies(file_path):
    """Improved IES file parser adhering to LM-63 standard."""
    with open(file_path, 'r') as file:
        lines = file.readlines()

    metadata = {}
    candela_values = []
    vertical_angles = []
    horizontal_angles = []

    parsing_candela = False
    vertical_angle_count, horizontal_angle_count = None, None

    for line in lines:
        line = line.strip()

        # Skip empty lines
        if not line:
            continue

        # Parse metadata
        if line.startswith("["):
            parts = line.strip('[]').split(' ', 1)
            if len(parts) == 2:
                key, value = parts
                metadata[key] = value
            continue

        # Parse tilt information and candela header
        if line.startswith("TILT"):
            metadata["tilt"] = line.split("=")[1].strip()
            continue

        # Parse the key photometric parameters
        if vertical_angle_count is None and len(line.split()) >= 10:
            parts = line.split()
            try:
                metadata["number_of_lamps"] = int(parts[0])
                metadata["lumens_per_lamp"] = float(parts[1])
                metadata["candela_multiplier"] = float(parts[2])
                vertical_angle_count = int(parts[3])
                horizontal_angle_count = int(parts[4])
                metadata["photometric_type"] = int(parts[5])
                metadata["units_type"] = int(parts[6])
                metadata["luminous_dimensions"] = tuple(map(float, parts[7:10]))
                print(f"Photometric Parameters: {metadata}")
                continue
            except ValueError:
                raise ValueError(f"Error parsing photometric parameters: {line}")

        # Parse vertical angles
        if vertical_angle_count is not None and len(vertical_angles) < vertical_angle_count:
            vertical_angles.extend(map(float, line.split()))
            if len(vertical_angles) > vertical_angle_count:
                # Trim excess values
                vertical_angles = vertical_angles[:vertical_angle_count]
            print(f"Vertical Angles ({len(vertical_angles)}/{vertical_angle_count}): {vertical_angles}")
            continue

        # Parse horizontal angles
        if horizontal_angle_count is not None and len(horizontal_angles) < horizontal_angle_count:
            horizontal_angles.extend(map(float, line.split()))
            if len(horizontal_angles) > horizontal_angle_count:
                # Trim excess values
                horizontal_angles = horizontal_angles[:horizontal_angle_count]
            print(f"Horizontal Angles ({len(horizontal_angles)}/{horizontal_angle_count}): {horizontal_angles}")
            continue

        # Parse candela values
        if len(vertical_angles) == vertical_angle_count and len(horizontal_angles) == horizontal_angle_count:
            if not parsing_candela:
                parsing_candela = True
                print("Switching to parsing candela values...")
            candela_values.extend(map(float, line.split()))
            print(f"Current Candela Values Count: {len(candela_values)}")


    # Validate parsed data
    if vertical_angle_count is None or horizontal_angle_count is None:
        raise ValueError("Missing vertical or horizontal angle count in the IES file.")

    expected_candela_count = vertical_angle_count * horizontal_angle_count
    if len(candela_values) != expected_candela_count:
        print(f"Expected Candela Count: {expected_candela_count}, Parsed: {len(candela_values)}")
        raise ValueError(f"Mismatch in candela data. Expected {expected_candela_count}, got {len(candela_values)}.")

    candela_matrix = np.array(candela_values).reshape((horizontal_angle_count, vertical_angle_count))

    return {
        'metadata': metadata,
        'vertical_angles': np.array(vertical_angles),
        'horizontal_angles': np.array(horizontal_angles),
        'candela_values': candela_matrix
    }

# Step 2: Compute illuminance at a given point
def compute_illuminance(photometric_data, point, height, pole_position):
    """
    Calculate illuminance at a given point on the ground (horizontal or vertical).

    Parameters:
        photometric_data: Parsed IES data
        point: Tuple (x, y, z) representing the point
        height: Height of the light source
        pole_position: Tuple (x, y) representing the light pole's position

    Returns:
        illuminance (lux)
    """
    # Adjust point relative to the pole's position
    x, y, z = point
    px, py = pole_position
    dx, dy = x - px, y - py

    # Calculate distance and angles
    r = np.sqrt(dx**2 + dy**2 + (z - height)**2)
    if r <= 0:
        raise ValueError(f"Invalid distance r={r}, computed from point {point} and pole {pole_position}.")
    vertical_angle = np.degrees(np.arctan2(np.sqrt(dx**2 + dy**2), height - z))
    if not (photometric_data['vertical_angles'][0] <= vertical_angle <= photometric_data['vertical_angles'][-1]):
        print(f"Vertical angle {vertical_angle} is out of range ({photometric_data['vertical_angles'][0]}, {photometric_data['vertical_angles'][-1]}).")
    #print('compute_illuminance : vertical_angle = ', vertical_angle)
    azimuth_angle = np.degrees(np.arctan2(dy, dx)) % 360

    # Interpolate luminous intensity for the given angle
    candelas = photometric_data['candela_values']
    vertical_angles = photometric_data['vertical_angles']

    # Select the plane closest to 0 degrees azimuth for simplicity
    intensity = interp1d(vertical_angles, candelas[0], kind='linear', fill_value="extrapolate")
    luminous_intensity = intensity(vertical_angle)

    illuminance = luminous_intensity / (r**2)
    return illuminance

# Step 3: Compute luminance to driver's eye
def compute_luminance(photometric_data, observer_position, pole_position, height, observer_angle=0, fov=30):
    """
    Calculate luminance perceived by the driver's eye, considering field of view (FOV).

    Parameters:
        photometric_data: Parsed IES data
        observer_position: Tuple (x, y, z) of the driver's eye
        pole_position: Tuple (x, y) of the light pole
        height: Height of the light source
        observer_angle: Observer's azimuth angle in degrees (default: 0 degrees)
        fov: Field of view in degrees (default: 30 degrees)

    Returns:
        luminance (cd/m^2)
    """
    ox, oy, oz = observer_position
    px, py = pole_position
    dx, dy = px - ox, py - oy

    # Compute azimuth angle between observer and pole
    azimuth_angle = (np.degrees(np.arctan2(dy, dx)) - observer_angle) % 360

    # Check if the pole is within the FOV
    if -fov / 2 <= azimuth_angle <= fov / 2 or 360 - fov / 2 <= azimuth_angle <= 360 + fov / 2:
        return compute_illuminance(photometric_data, observer_position, height, pole_position) / np.pi
    else:
        return 0  # Outside FOV
    
    
# Step 4: Visualization

def plot_combined_illuminance_environment(photometric_data, height, crosswalk_position=(0, 0), pole_position=(0, 0), observer_position=(3, 0, 1.2), observer_angle=0, fov=30, grid_size=10, resolution=100):
    """
    Plot combined horizontal illuminance distribution and environment layout.

    Parameters:
        photometric_data: Parsed IES data
        height: Height of the light pole
        crosswalk_position: (x, y) position of the crosswalk center
        pole_position: (x, y) position of the light pole
        observer_angle: Observer's azimuth angle in degrees (default: 0 degrees)
        grid_size: Size of the grid for illuminance distribution
        resolution: Resolution of the grid
    """
    x = np.linspace(-grid_size, grid_size, resolution)
    y = np.linspace(-grid_size, grid_size, resolution)
    X, Y = np.meshgrid(x, y)

    illuminance = np.zeros_like(X)
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            #illuminance[i, j] = compute_illuminance(photometric_data, (X[i, j], Y[i, j], 0), height)
            illuminance[i, j] = compute_illuminance(photometric_data, (X[i, j], Y[i, j], 0), height, pole_position)

    plt.figure(figsize=(10, 8))

    # Contour plot for illuminance
    contour = plt.contourf(X, Y, illuminance, levels=100, cmap='inferno', alpha=0.7)
    plt.colorbar(contour, label='Illuminance (lux)')

    # Plot the light pole
    plt.scatter(*pole_position, color='blue', label='Light Pole', s=100)
    plt.text(pole_position[0] + 0.5, pole_position[1] + 0.5, 'Light Pole', fontsize=10)

    # Plot the crosswalk
    crosswalk_size = 1  # Width of the crosswalk in meters
    crosswalk = plt.Rectangle((crosswalk_position[0] - crosswalk_size / 2, crosswalk_position[1] - crosswalk_size / 2),
                               crosswalk_size, crosswalk_size, color='red', alpha=0.5, label='Crosswalk (Calculated point)')
    plt.gca().add_patch(crosswalk)
    
    # Annotate illuminance at the crosswalk
    crosswalk_illuminance = compute_illuminance(photometric_data, (*crosswalk_position, 0),height, pole_position)
    plt.text(crosswalk_position[0] - 2.5, crosswalk_position[1] - 0.5,
             f'Illuminance: {crosswalk_illuminance:.2f} lux', fontsize=10, color='red')

    # Plot the observer (driver position)
    plt.scatter(observer_position[0], observer_position[1], color='green', label='Observer (Driver)', s=100)
    plt.text(observer_position[0] + 0.5, observer_position[1] + 0.5, f'Driver\n({observer_position[0]:.1f}, {observer_position[1]:.1f}, {observer_position[2]:.1f} m)', fontsize=10, color='green')

    # Add observer's FOV
    fov_angle1 = np.radians(observer_angle - fov / 2)
    fov_angle2 = np.radians(observer_angle + fov / 2)
    fov_line1 = (observer_position[0] + np.cos(fov_angle1) * grid_size,
                 observer_position[1] + np.sin(fov_angle1) * grid_size)
    fov_line2 = (observer_position[0] + np.cos(fov_angle2) * grid_size,
                 observer_position[1] + np.sin(fov_angle2) * grid_size)
    plt.plot([observer_position[0], fov_line1[0]], [observer_position[1], fov_line1[1]], color='green', linestyle='--', alpha=0.7)
    plt.plot([observer_position[0], fov_line2[0]], [observer_position[1], fov_line2[1]], color='green', linestyle='--', alpha=0.7)
    
    # Add luminance annotation
    luminance = compute_luminance(photometric_data, observer_position, pole_position, height, fov=30)
    plt.text(observer_position[0] + 0.5, observer_position[1] - 0.5, f'Luminance: {luminance:.2f} cd/m^2', fontsize=10, color='black')
    
    # Add grid and labels
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.axhline(0, color='black', linewidth=0.5)
    plt.axvline(0, color='black', linewidth=0.5)
    plt.xlabel('X (m)')
    plt.ylabel('Y (m)')
    plt.legend()
    plt.title('Combined Illuminance Distribution and Environment Layout')
    plt.show()

# Step 5: Run Simulation
def main():
    # Example IES file path
    ies_file = "Luminaires\IES-LM-63\Circular Downlight 8-inch 85W 5840 lm.ies"
    #ies_file = "Luminaires\IES-LM-63\Linear Troffer Parabolic 2x4' 64W 3980 lm.ies"
    
    # Load photometric data
    photometric_data = load_ies(ies_file)

    # Define parameters
    luminaire_height = 6  # in meters
    observer_position = (6, 0, 1.2)  # Driver's eye position (x, y, z)
    observer_angle = 180
    fov = 90
    crosswalk_position = (0, 0)  # Center of the crosswalk
    pole_position = (2, 2)  # Position of the light pole

    # Compute illuminance and luminance
    horizontal_illuminance = compute_illuminance(photometric_data, (*crosswalk_position, 0), luminaire_height, pole_position)
    luminance = compute_luminance(photometric_data, observer_position, pole_position, luminaire_height,  observer_angle=observer_angle, fov=fov)

    print(f"Horizontal Illuminance at Crosswalk Center: {horizontal_illuminance:.2f} lux")
    print(f"Luminance to Driver's Eye: {luminance:.2f} cd/m^2")

    # Plot combined visualization
    plot_combined_illuminance_environment(photometric_data, luminaire_height, crosswalk_position, pole_position, observer_position,observer_angle=observer_angle, fov=fov)

if __name__ == "__main__":
    main()
