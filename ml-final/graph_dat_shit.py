import matplotlib.pyplot as plt
import csv

def plot_loss_array(data, title="Loss Over Time", filename="loss_data.csv"):
    if data is None:
        raise ValueError("Loss data is None. Ensure your training function returns proper loss values.")

    full_list = []
    epoch_boundaries = []

    # Flatten data and track where each epoch ends
    for epoch_losses in data:
        full_list.extend(epoch_losses)
        epoch_boundaries.append(len(full_list))  # mark where each epoch ends

    # Plotting
    plt.figure(figsize=(10, 6))
    plt.plot(full_list, linestyle='-', marker='o')

    for epoch_idx, boundary in enumerate(epoch_boundaries):
        plt.axvline(boundary, color='r', linestyle='--', label=f"End Epoch {epoch_idx + 1}")
        plt.text(boundary + 0.1, full_list[boundary - 1], f'Epoch {epoch_idx + 1}',
                 color='r', verticalalignment='bottom')

    plt.title(title)
    plt.xlabel("Logged Step")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Save as CSV
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Global Step", "Loss"])
        for i, loss in enumerate(full_list):
            writer.writerow([i + 1, loss])

    print(f"Loss data saved to {filename}")
