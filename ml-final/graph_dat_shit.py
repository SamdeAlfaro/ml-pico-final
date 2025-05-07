import matplotlib.pyplot as plt

def plot_loss_array(data, labels=None, title="Three Arrays Plot"):
    full_list = []
    index_of_epoch = []
    for it,i in enumerate(data):
        for x in i:
            full_list.append(x)
        if it%2 !=0:
            index_of_epoch.append(i)
    print(full_list)
    
    plt.figure(figsize=(10, 6))
    plt.plot(full_list, linestyle='-', marker='o')
    for it, idx in enumerate(index_of_epoch):
        plt.axvline(full_list.index(idx[0]), color='r', linestyle='--', label=f'Epoch {it+1} Avg = {format(idx[0], ".2f")}')
        plt.text(full_list.index(idx[0])+.1,idx[0]+.1, f'Epoch {it+1}', color='r', rotation=0, verticalalignment='top')
    plt.title(title)
    plt.xlabel("Time / Number of Steps")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()
