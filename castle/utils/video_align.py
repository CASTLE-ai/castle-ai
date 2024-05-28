import cv2
import numpy as np

    
def get_centroids(frame, roi):
    connected_components = roi_connected_components(frame, roi)
    num_labels, _, stats, centroids = connected_components
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    max_label = np.argmax(areas)
    x = centroids[max_label + 1][0]
    y = centroids[max_label + 1][1]
    return x, y


def center_roi(frame, mask, roi_color):
    h, w = frame.shape[:2]
    center = (w // 2, h // 2)
    roi_x, roi_y = get_centroids(mask, roi_color)
    matrix = np.float32([[1, 0, center[0] - roi_x], [0, 1, center[1] - roi_y]])
    return cv2.warpAffine(frame, matrix, (w, h))



def rotate_based_on_roi_closest_center_point(frame, mask, roi_color):
    roi_contour = get_contour(mask, roi_color)
    h, w = frame.shape[:2]
    center_x, conter_y = (w // 2, h // 2)
    (closest_point_x, closest_point_y), _ = find_closest_point((center_x, conter_y), roi_contour)
    theta = np.arctan2(closest_point_y - conter_y, closest_point_x - center_x) * 180. / np.pi
    matrix = cv2.getRotationMatrix2D((center_x, conter_y), theta-90, 1.0)
    return cv2.warpAffine(frame, matrix, (w, h))



def crop(frame, crop_h, crop_w):
    h, w = frame.shape[:2]
    u, d = int(h // 2 - crop_h // 2), int(h // 2 + crop_h // 2)
    l, r = int(w // 2 - crop_w // 2), int(w // 2 + crop_w // 2)
    return frame[u:d, l:r]





def roi_connected_components(frame, select_roi, tolerance=30):

    if type(select_roi) == list and len(select_roi) == 3:
        h, w = frame.shape[:2]
        mask = np.zeros((h, w)).astype(np.uint8)
        lower_bound = np.array([select_roi[0] - tolerance,
                                select_roi[1] - tolerance,
                                select_roi[2] - tolerance])
        upper_bound = np.array([select_roi[0] + tolerance,
                                select_roi[1] + tolerance,
                                select_roi[2] + tolerance])
        within_range = cv2.inRange(frame, lower_bound, upper_bound)
        mask[within_range > 0] = 255
    else:
        select_roi = int(select_roi)

        mask = cv2.inRange(frame, select_roi, select_roi)


    
    output = cv2.connectedComponentsWithStats(mask, 8, cv2.CV_32S)
    num_labels = output[0]
    assert num_labels > 1, 'roi_connected_components error'
    return output



def get_contour(frame, roi):
    connected_components = roi_connected_components(frame, roi)
    num_labels, labels, stats, centroids = connected_components
    
    areas = [stats[j, cv2.CC_STAT_AREA] for j in range(1, num_labels)]
    maxi_comp_id = np.argmax(areas)
    selected_label = (labels == (maxi_comp_id+1)).astype(np.uint8) * 255
    _, binary_mask = cv2.threshold(selected_label, 0, 255, cv2.THRESH_BINARY)
    contour = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0][0]
    contour = contour.squeeze()
    assert contour.ndim > 1, 'get_contour error'
    return contour
    



def get_mask(frame, select_roi):
    connected_components = roi_connected_components(frame, select_roi)
    num_labels, labels, stats, centroids = connected_components
    areas = [stats[i, cv2.CC_STAT_AREA] for i in range(1, num_labels)]
    max_label = np.argmax(areas)
    return (labels == (max_label+1))
    
    
def find_closest_point(ref, contour):
    mini = int(1e6)
    point_close = None
    assert len(contour) >= 2, 'find_closest_point error'
    for i in range(len(contour)):
        distance = cv2.norm(ref - contour[i])
        if distance < mini:
            mini = distance
            point_close = contour[i]

    return point_close, mini
    

def blank_page(crop_h, crop_w):
    white_frame = np.full((crop_h, crop_w, 3), 255, dtype=np.uint8)
    return white_frame
    
