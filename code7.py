import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial import distance as dist
from scipy.optimize import linear_sum_assignment
import matplotlib.colors as colors
from collections import defaultdict
import seaborn as sns  # For KDE plots
import pandas as pd  # For saving results
from scipy.stats import gaussian_kde

# === 1. Calibration Parameters ===

# Spatial Calibration: pixels to real-world units (e.g., micrometers per pixel)
PIXEL_TO_MICROMETER = 5  # Adjust this based on your calibration

# Temporal Calibration: frame rate of the video
FRAME_RATE = 30  # frames per second

# === 2. Particle Tracking Using Optical Flow ===

def sample_frames(video_path, num_frames=200, skip_interval=2):
    """
    Samples frames from the video at specified intervals.

    Parameters:
    - video_path: Path to the video file.
    - num_frames: Number of frames to sample.
    - skip_interval: Number of frames to skip between samples.

    Returns:
    - List of grayscale frames.
    """
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0
    while len(frames) < num_frames and cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        if count % skip_interval == 0:
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            frames.append(gray_frame)
        count += 1
    cap.release()
    return frames

def preprocess_frame(frame):
    """
    Preprocesses the frame using adaptive thresholding and morphological operations.

    Parameters:
    - frame: Grayscale image.

    Returns:
    - Preprocessed binary image.
    """
    # Adaptive Thresholding
    thresh = cv2.adaptiveThreshold(
        frame, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 15, 1
    )
    # Morphological Operations
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    clean = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    clean = cv2.morphologyEx(clean, cv2.MORPH_DILATE, kernel, iterations=1)
    return clean

def get_blob_detector():
    """
    Configures and returns a SimpleBlobDetector with specified parameters.

    Returns:
    - Configured SimpleBlobDetector object.
    """
    params = cv2.SimpleBlobDetector_Params()
    
    # Area filtering
    params.filterByArea = True
    params.minArea = 2  # Lowered to detect smaller particles
    params.maxArea = 100  # Increased to detect larger particles if any
    
    # Circularity filtering
    params.filterByCircularity = True
    params.minCircularity = 0.2  # Lowered to allow less circular particles
    
    # Inertia filtering
    params.filterByInertia = True
    params.minInertiaRatio = 0.1  # Lowered to allow more elongated particles
    
    # Convexity filtering
    params.filterByConvexity = True
    params.minConvexity = 0.4  # Lowered to allow less convex particles
    
    # Create the detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)
    return detector

def detect_particles(frame, detector):
    """
    Detects particles in the frame using the provided blob detector.

    Parameters:
    - frame: Preprocessed binary image.
    - detector: Configured SimpleBlobDetector object.

    Returns:
    - List of detected keypoints.
    """
    keypoints = detector.detect(frame)
    return keypoints

def calculate_sizes(keypoints):
    """
    Calculates the physical sizes of detected particles.

    Parameters:
    - keypoints: List of detected keypoints.

    Returns:
    - List of particle sizes in micrometers.
    """
    sizes = [kp.size * PIXEL_TO_MICROMETER for kp in keypoints]  # Convert to micrometers
    return sizes

def calculate_density(keypoints, frame_area):
    """
    Calculates particle density per frame.

    Parameters:
    - keypoints: List of detected keypoints.
    - frame_area: Total area of the frame in pixels.

    Returns:
    - Density in particles/cm².
    """
    density = len(keypoints) / frame_area  # particles per pixel
    density_per_cm2 = density / (PIXEL_TO_MICROMETER**2) * 1e8  # Convert to particles/cm²
    return density_per_cm2

# === 3. Visualization Enhancements ===

def plot_distributions(all_sizes, all_velocities, all_densities):
    """
    Plots normalized histograms for size and velocity distributions, and density over time.

    Parameters:
    - all_sizes: List of particle sizes.
    - all_velocities: List of particle velocities.
    - all_densities: List of densities per frame.
    """
    plt.figure(figsize=(18, 5))

    # Size Distribution
    plt.subplot(1, 3, 1)
    plt.hist(all_sizes, bins=50, color='blue', alpha=0.7, density=True)
    plt.title('Particle Size Distribution')
    plt.xlabel('Size (micrometers)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    
    # Set x-axis ticks with a gap of 2 micrometers
    if all_sizes:
        min_size = 0
        max_size = max(all_sizes)
        plt.xticks(np.arange(min_size, max_size + 5, 5))
    else:
        plt.xticks([])  # No ticks if no data

    # Velocity Distribution
    plt.subplot(1, 3, 2)
    plt.hist(all_velocities, bins=50, color='green', alpha=0.7, density=True)
    plt.title('Particle Speed Distribution')
    plt.xlabel('Speed (µm/s)')
    plt.ylabel('Probability Density')
    plt.grid(True)
    plt.xticks()  # Default ticks

    # Density Over Time
    plt.subplot(1, 3, 3)
    plt.plot(all_densities, color='red')
    plt.title('Particle Density Over Time')
    plt.xlabel('Frame Number')
    plt.ylabel('Density (particles/cm²)')
    plt.grid(True)
    plt.xticks()  # Default ticks

    plt.tight_layout()
    plt.show()

def plot_heatmap(preprocessed_frames, all_keypoints):
    """
    Plots a heatmap showing cumulative particle density across all frames.

    Parameters:
    - preprocessed_frames: List of preprocessed binary frames.
    - all_keypoints: List of keypoints detected in all frames.
    """
    # Create an empty array for heatmap
    heatmap = np.zeros_like(preprocessed_frames[0], dtype=np.float32)

    for keypoints in all_keypoints:
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            if 0 <= y < heatmap.shape[0] and 0 <= x < heatmap.shape[1]:
                heatmap[y, x] += 1

    total_particles = np.sum(heatmap)
    print(f"Total particles contributing to heatmap: {total_particles}")

    if total_particles == 0:
        print("No particles to display in heatmap.")
        return

    # Apply Gaussian Blur for smoother visualization
    heatmap = cv2.GaussianBlur(heatmap, (15, 15), 0)

    # Normalize heatmap
    heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
    heatmap = heatmap.astype(np.uint8)

    plt.figure(figsize=(6,6))
    plt.imshow(heatmap, cmap='viridis')  # Changed colormap for better visibility
    plt.title('Particle Density Heatmap with Gaussian Blur')
    plt.axis('off')
    plt.show()

def plot_spatial_density_kde_seaborn(all_positions, frame_shape):
    """
    Plots a spatial density representation using Seaborn's KDE plot with a scale.
    
    Parameters:
    - all_positions: List of (x, y) tuples representing particle positions.
    - frame_shape: Tuple representing the frame dimensions (height, width).
    """
    if not all_positions:
        print("No positions to plot.")
        return

    # Separate x and y positions and convert to micrometers
    x_pixels = [pos[0] for pos in all_positions]
    y_pixels = [pos[1] for pos in all_positions]
    x_microm = [x * PIXEL_TO_MICROMETER for x in x_pixels]
    y_microm = [y * PIXEL_TO_MICROMETER for y in y_pixels]

    plt.figure(figsize=(8,6))
    
    # Use Seaborn's kdeplot with adjusted bandwidth and fill
    kde = sns.kdeplot(x=x_microm, y=y_microm, cmap="Reds", fill=True, bw_adjust=1)
    
    plt.title('Spatial Density Representation (X-Y) using KDE\nRed indicates higher density')
    plt.xlabel('X Position (µm)')
    plt.ylabel('Y Position (µm)')
    plt.gca().invert_yaxis()  # Invert Y-axis to match image coordinates

    # Add a scale bar
    add_scale_bar(plt.gca(), length=150, location=(0.1, 0.9))  # Example: 150 µm scale bar

    # Add colorbar manually using the QuadContourSet's first collection
    plt.colorbar(kde.collections[0], label='Density')
    
    plt.show()

def add_scale_bar(ax, length, location=(0.1, 0.9), linewidth=3):
    """
    Adds a scale bar to the plot.

    Parameters:
    - ax: Matplotlib axis object.
    - length: Length of the scale bar in micrometers.
    - location: Tuple indicating the relative location (x, y) in the plot [0,1].
    - linewidth: Thickness of the scale bar.
    """
    x0, y0 = location
    x_min, x_max = ax.get_xlim()
    y_min, y_max = ax.get_ylim()

    # Calculate scale bar position in data coordinates
    sb_x = x_min + (x_max - x_min) * x0
    sb_y = y_min + (y_max - y_min) * y0

    # Draw scale bar
    ax.plot([sb_x, sb_x + length], [sb_y, sb_y], color='black', linewidth=linewidth)
    ax.text(sb_x + length / 2, sb_y + (y_max - y_min)*0.02, f'{length} µm',
            horizontalalignment='center', verticalalignment='bottom')

def plot_trajectories(tracker):
    """
    Plots the trajectories of tracked particles.

    Parameters:
    - tracker: SimpleTracker object containing track data.
    """
    plt.figure(figsize=(8,6))
    for tid, data in tracker.tracks.items():
        positions = data['positions']
        if len(positions) > 1:
            x = [pos[0] * PIXEL_TO_MICROMETER for pos in positions]
            y = [pos[1] * PIXEL_TO_MICROMETER for pos in positions]
            plt.plot(x, y, label=f'ID {tid}')
    plt.title('Particle Trajectories')
    plt.xlabel('X Position (µm)')
    plt.ylabel('Y Position (µm)')
    plt.gca().invert_yaxis()  # Match image coordinates
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_velocity_color_coded(all_sizes, all_velocities):
    """
    Plots a color-coded scatter plot of particle size vs velocity.

    Parameters:
    - all_sizes: List of particle sizes.
    - all_velocities: List of particle velocities.
    """
    plt.figure(figsize=(8,6))
    scatter = plt.scatter(all_sizes, all_velocities, c=all_velocities, cmap='viridis', alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.title('Particle Size vs Velocity (Color-Coded)')
    plt.xlabel('Size (micrometers)')
    plt.ylabel('Velocity (µm/s)')
    cbar = plt.colorbar(scatter)
    cbar.set_label('Velocity (µm/s)')
    plt.grid(True)
    plt.show()

def plot_size_vs_velocity(all_sizes, all_velocities):
    """
    Plots a scatter plot of particle size vs velocity.

    Parameters:
    - all_sizes: List of particle sizes.
    - all_velocities: List of particle velocities.
    """
    plt.figure(figsize=(8,6))
    plt.scatter(all_sizes, all_velocities, alpha=0.7, edgecolors='w', linewidth=0.5)
    plt.title('Particle Size vs Velocity')
    plt.xlabel('Size (micrometers)')
    plt.ylabel('Velocity (µm/s)')
    plt.grid(True)
    plt.show()

# === 4. Simple Tracker Implementation ===

class SimpleTracker:
    def __init__(self, max_distance=80, max_lost=5):
        """
        Initializes the SimpleTracker.

        Parameters:
        - max_distance: Maximum distance to associate detections with existing tracks.
        - max_lost: Maximum number of frames to keep "lost" tracks before termination.
        """
        self.next_id = 0
        self.tracks = dict()  # id: {'positions': [], 'velocities': [], 'lost': 0}
        self.max_distance = max_distance
        self.max_lost = max_lost  # Maximum number of frames to keep "lost" tracks

    def update(self, detections):
        """
        Updates tracks with current detections.

        Parameters:
        - detections: List of (x, y) tuples.

        Returns:
        - matched: List of tuples (track_id, (x, y))
        """
        matched = []

        if len(self.tracks) == 0:
            for det in detections:
                self.tracks[self.next_id] = {'positions': [det], 'velocities': [0.0], 'lost': 0}
                matched.append((self.next_id, det))
                self.next_id += 1
            return matched
        
        track_ids = list(self.tracks.keys())
        track_positions = [self.tracks[tid]['positions'][-1] for tid in track_ids]

        if len(detections) == 0:
            # Increment 'lost' counter for all tracks
            for tid in track_ids:
                self.tracks[tid]['lost'] += 1
            # Remove tracks that have exceeded max_lost
            self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.max_lost}
            return matched

        # Compute distance matrix
        D = dist.cdist(track_positions, detections)
        row_ind, col_ind = linear_sum_assignment(D)

        assigned_tracks = set()
        assigned_detections = set()

        for r, c in zip(row_ind, col_ind):
            if D[r, c] < self.max_distance:
                track_id = track_ids[r]
                det = detections[c]
                self.tracks[track_id]['positions'].append(det)
                self.tracks[track_id]['lost'] = 0  # Reset 'lost' counter

                # Calculate velocity if possible
                positions = self.tracks[track_id]['positions']
                if len(positions) >= 2:
                    dx = positions[-1][0] - positions[-2][0]
                    dy = positions[-1][1] - positions[-2][1]
                    distance_pixels = np.sqrt(dx**2 + dy**2)
                    distance_micrometers = distance_pixels * PIXEL_TO_MICROMETER
                    velocity = distance_micrometers * FRAME_RATE  # µm/s
                    self.tracks[track_id]['velocities'].append(velocity)
                else:
                    # No velocity can be calculated for the first position
                    self.tracks[track_id]['velocities'].append(0.0)  # Assign zero or np.nan as appropriate

                matched.append((track_id, det))
                assigned_tracks.add(track_id)
                assigned_detections.add(c)

        # Increment 'lost' counter for unmatched tracks
        for tid in track_ids:
            if tid not in assigned_tracks:
                self.tracks[tid]['lost'] += 1

        # Remove tracks that have exceeded max_lost
        self.tracks = {tid: t for tid, t in self.tracks.items() if t['lost'] <= self.max_lost}

        # Create new tracks for unmatched detections
        for c, det in enumerate(detections):
            if c not in assigned_detections:
                self.tracks[self.next_id] = {'positions': [det], 'velocities': [0.0], 'lost': 0}
                matched.append((self.next_id, det))
                self.next_id += 1

        return matched

    def get_previous_position(self, track_id):
        """
        Retrieves the previous position of a given track.

        Parameters:
        - track_id: The ID of the track.

        Returns:
        - Tuple of (previous_position, current_position) if available, else (None, current_position)
        """
        positions = self.tracks[track_id]['positions']
        if len(positions) >= 2:
            return positions[-2], positions[-1]
        elif len(positions) == 1:
            return None, positions[-1]
        else:
            return None, None

# === 5. Saving and Exporting Results ===

def save_results(all_sizes, all_velocities, all_densities, all_individual_velocities):
    """
    Saves analysis results to CSV files.

    Parameters:
    - all_sizes: List of particle sizes.
    - all_velocities: List of particle velocities.
    - all_densities: List of densities per frame.
    - all_individual_velocities: List of all individual velocity measurements.
    """
    # Create a DataFrame for particle sizes and velocities
    df = pd.DataFrame({
        'Size (µm)': all_sizes,
        'Velocity (µm/s)': all_individual_velocities
    })
    
    # Save to CSV
    df.to_csv('particle_analysis.csv', index=False)
    print("Particle analysis results saved to 'particle_analysis.csv'.")

    # Save density over time
    density_df = pd.DataFrame({
        'Frame Number': range(1, len(all_densities)+1),
        'Density (particles/cm²)': all_densities
    })
    density_df.to_csv('density_over_time.csv', index=False)
    print("Density over time results saved to 'density_over_time.csv'.")

    # Save individual velocities
    velocity_df = pd.DataFrame({
        'Velocity (µm/s)': all_individual_velocities
    })
    velocity_df.to_csv('individual_velocities.csv', index=False)
    print("Individual velocities saved to 'individual_velocities.csv'.")

# === 6. Main Processing ===

def main():
    video_path = 'VID-20240812-WA0006.mp4'  # Update with your video path

    # Load and preprocess frames
    print("Sampling frames...")
    sampled_frames = sample_frames(video_path, num_frames=200, skip_interval=2)
    print(f"Total sampled frames: {len(sampled_frames)}")

    print("Preprocessing frames...")
    preprocessed_frames = [preprocess_frame(frame) for frame in sampled_frames]

    frame_area = (preprocessed_frames[0].shape[0] * preprocessed_frames[0].shape[1])

    # Initialize Tracker
    tracker = SimpleTracker(max_distance=100, max_lost=5)  # Adjusted max_distance and max_lost

    # Lists to store all measurements
    all_sizes = []
    all_velocities = []
    all_densities = []
    all_keypoints = []
    all_positions = []  # For spatial density representation

    for i, frame in enumerate(preprocessed_frames):
        print(f"Processing frame {i+1}/{len(preprocessed_frames)}...")
        # Create a new blob detector for each frame
        blob_detector = get_blob_detector()
        keypoints = detect_particles(frame, blob_detector)
        sizes = calculate_sizes(keypoints)
        density = calculate_density(keypoints, frame_area)
        
        all_sizes.extend(sizes)
        all_densities.append(density)
        all_keypoints.append(keypoints)
        
        # Extract positions
        detections = [kp.pt for kp in keypoints]
        all_positions.extend(detections)
        
        # Debugging: Print number of detections
        print(f"Detected {len(keypoints)} particles in frame {i+1}")
        
        # Update tracker
        matched = tracker.update(detections)
        
        # Prepare frame for drawing (convert grayscale to BGR)
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        
        # Draw detected particles
        for kp in keypoints:
            x, y = int(kp.pt[0]), int(kp.pt[1])
            cv2.circle(frame_bgr, (x, y), int(kp.size / 2), (0, 0, 255), 2)  # Red circles
        
        # Draw velocity vectors and labels
        for track_id, det in matched:
            prev_pos, curr_pos = tracker.get_previous_position(track_id)
            if prev_pos is not None:
                prev_x, prev_y = int(prev_pos[0]), int(prev_pos[1])
                curr_x, curr_y = int(curr_pos[0]), int(curr_pos[1])
                # Draw an arrow from previous position to current position
                cv2.arrowedLine(
                    frame_bgr,
                    (prev_x, prev_y),
                    (curr_x, curr_y),
                    color=(255, 0, 0),  # Blue color in BGR
                    thickness=2,
                    tipLength=0.3
                )
                # Retrieve the latest velocity
                velocity = tracker.tracks[track_id]['velocities'][-1]
                # Label the track ID and velocity near the current position
                label = f'Z_({track_id}, {velocity:.2f})'
                cv2.putText(
                    frame_bgr,
                    label,
                    (curr_x + 5, curr_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),  # Green color in BGR
                    1
                )
            else:
                # For the first detection of a track, label only with ID
                curr_x, curr_y = int(curr_pos[0]), int(curr_pos[1])
                label = f'Z_{track_id}'
                cv2.putText(
                    frame_bgr,
                    label,
                    (curr_x + 5, curr_y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.4,
                    (0, 255, 0),  # Green color in BGR
                    1
                )
        
        # Optional: Visualize detection and velocity vectors on some frames
        if (i+1) % 50 == 0 or i == 0:
            # Convert BGR to RGB for correct color display in Matplotlib
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            plt.figure(figsize=(6,6))
            plt.imshow(frame_rgb)
            plt.title(f"Particle Detection & Velocity - Frame {i+1}")
            plt.axis('off')
            plt.show()

    # Calculate all individual velocities
    print("Calculating individual velocities...")
    all_individual_velocities = []
    for track in tracker.tracks.values():
        all_individual_velocities.extend(track['velocities'])

    # Debugging: Verify velocity data
    print(f"Total number of individual velocity measurements: {len(all_individual_velocities)}")
    if len(all_individual_velocities) > 0:
        print("Sample of individual velocities (µm/s):")
        sample_velocities = all_individual_velocities[:10]  # Display first 10 for brevity
        print(sample_velocities)
    else:
        print("No velocity measurements available.")

    # Debugging: Verify size data
    print(f"Total number of size measurements: {len(all_sizes)}")
    if len(all_sizes) > 0:
        print("Sample sizes (µm):", all_sizes[:10])  # Print first 10 sizes
    else:
        print("No size measurements available.")

    # Filter out unrealistic velocities (optional)
    # Example: Remove velocities outside 0 - 100000 µm/s (adjust as needed)
    all_velocities_filtered = [v for v in all_individual_velocities if 0 <= v <= 100000]
    print(f"Number of velocities after filtering: {len(all_velocities_filtered)}")

    # Visualization
    print("Plotting distributions...")
    plot_distributions(all_sizes, all_velocities_filtered, all_densities)

    print("Plotting spatial density representation (X-Y) as Heatmap...")
    plot_heatmap(preprocessed_frames, all_keypoints)

    print("Plotting spatial density representation (X-Y) using Seaborn KDE...")
    plot_spatial_density_kde_seaborn(all_positions, preprocessed_frames[0].shape)

    print("Plotting size vs velocity...")
    plot_size_vs_velocity(all_sizes, all_velocities_filtered)

    print("Plotting particle trajectories...")
    plot_trajectories(tracker)

    print("Plotting color-coded velocity scatter plot...")
    plot_velocity_color_coded(all_sizes, all_velocities_filtered)

    # Save the results
    print("Saving analysis results...")
    save_results(all_sizes, all_velocities_filtered, all_densities, all_individual_velocities)

    # Display summary statistics
    if len(all_velocities_filtered) > 0:
        mean_velocity = np.mean(all_velocities_filtered)
        median_velocity = np.median(all_velocities_filtered)
        std_velocity = np.std(all_velocities_filtered)
        print(f"\nVelocity Statistics:")
        print(f"Mean Velocity: {mean_velocity:.2f} µm/s")
        print(f"Median Velocity: {median_velocity:.2f} µm/s")
        print(f"Standard Deviation: {std_velocity:.2f} µm/s")
    else:
        print("No velocity measurements to display statistics.")

if __name__ == "__main__":
    main()
