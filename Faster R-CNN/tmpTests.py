import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

def visTargets(images, targets):
    for image, target in zip(images, targets):
        image = image.permute(1, 2, 0).cpu().numpy()  # Convert image tensor to numpy array
        image = (image * 255).astype('uint8')  # Scale the image values to 0-255

        fig, ax = plt.subplots(1)
        ax.imshow(image)

        for box, cls in zip(target['boxes'], target['labels']):
            xmin, ymin, xmax, ymax = box.tolist()
            bbox = patches.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, linewidth=2, edgecolor='g', facecolor='none')
            ax.add_patch(bbox)
            ax.text(xmin, ymin - 10, f'Class: {cls}', bbox=dict(facecolor='g', edgecolor='g', alpha=0.5), fontsize=9, color='w')

        plt.axis('off')
        plt.show()
