import numpy as np
from sklearn.neighbors import LocalOutlierFactor
from scipy.stats import zscore

def data_filter(caption_data, pseudo_labels, similarities, beta):
    labeled_data = {}

    for i in range(len(caption_data)):
        label = pseudo_labels[i]
        if label != -1:
            if label not in labeled_data:
                labeled_data[label] = []
            labeled_data[label].append(caption_data[i])

    sorted_labeled_data = {k: labeled_data[k] for k in sorted(labeled_data.keys())}

    filtered_data = {}

    for label, image_text_pairs in sorted_labeled_data.items():
        sim_array = similarities[label].cpu().numpy()

        q1 = np.percentile(sim_array, 25)
        q3 = np.percentile(sim_array, 75)
        iqr = q3 - q1
        lower_bound = q1 - beta * iqr
        outlier_indices = [i for i, sim in enumerate(sim_array) if sim < lower_bound]
        filtered_image_text_pairs = [pair for i, pair in enumerate(image_text_pairs) if i not in outlier_indices]
        filtered_data[label] = filtered_image_text_pairs

    return filtered_data






