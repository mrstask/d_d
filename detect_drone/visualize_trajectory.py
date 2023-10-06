import csv
import matplotlib.pyplot as plt

from detect_drone.settings import OUTPUT_DIR


def read_trajectory_data(filename=f'{OUTPUT_DIR}/trajectory_data.csv'):
    data = []
    with open(filename, newline='') as file:
        reader = csv.reader(file)
        next(reader, None)  # Skip the header
        for row in reader:
            frame, x, y, w, h = map(int, row)  # Assuming all values are integers
            center_x = x + w / 2
            center_y = y + h / 2
            data.append((frame, center_x, center_y))
    return data


def plot_trajectory(data):
    frames, x_coords, y_coords = zip(*data)

    fig, ax = plt.subplots()
    ax.plot(frames, x_coords, label='X Coordinate')
    ax.plot(frames, y_coords, label='Y Coordinate')

    ax.set_xlabel('Frame')
    ax.set_ylabel('Position')
    ax.set_title('Drone Trajectory Over Time')
    ax.legend()

    plt.show()


if __name__ == '__main__':
    trajectory_data = read_trajectory_data()
    plot_trajectory(trajectory_data)
