import os
import time
import torch
from models.common import DetectMultiBackend
from utils.datasets import LoadImages
from utils.general import non_max_suppression, scale_coords
from utils.torch_utils import select_device
from ensemble_boxes_wbf import *


def iou(bbox1, bbox2):
    bbox1 = [float(x) for x in bbox1]
    bbox2 = [float(x) for x in bbox2]
    (x0_1, y0_1, x1_1, y1_1) = bbox1
    (x0_2, y0_2, x1_2, y1_2) = bbox2
    overlap_x0 = max(x0_1, x0_2)
    overlap_y0 = max(y0_1, y0_2)
    overlap_x1 = min(x1_1, x1_2)
    overlap_y1 = min(y1_1, y1_2)
    if overlap_x1 - overlap_x0 <= 0 or overlap_y1 - overlap_y0 <= 0:
        return 0
    size_1 = (x1_1 - x0_1) * (y1_1 - y0_1)
    size_2 = (x1_2 - x0_2) * (y1_2 - y0_2)
    size_intersection = (overlap_x1 - overlap_x0) * (overlap_y1 - overlap_y0)
    size_union = size_1 + size_2 - size_intersection
    return size_intersection / size_union


def track_iou(detections, tracks_active, sigma_l, sigma_h, sigma_iou, t_life, t_loss, id):
    tracks_finished = []
    dets_1 = [det for det in detections if det['score'] >= sigma_l and det['cls'] == 1]
    dets_2 = [det for det in detections if det['score'] >= sigma_l and det['cls'] == 2]
    updated_tracks = []
    for track in tracks_active:
        if track['cls'] == 1:
            if len(dets_1) > 0:
                best_match = max(dets_1, key=lambda x: iou(track['bboxes'], x['bbox']))
                if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                    track['bboxes'] = best_match['bbox']
                    track['max_score'] = max(track['max_score'], best_match['score'])
                    track['life_time'] += 1
                    track['loss_time'] = 0
                    updated_tracks.append(track)
                    del dets_1[dets_1.index(best_match)]
                    if track['max_score'] >= sigma_h and track['life_time'] >= t_life:
                        if track['id'] == 0:
                            id += 1
                            track['id'] = id
                        tracks_finished.append(track)
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                track['loss_time'] += 1
                if track['loss_time'] > 5:
                    track['life_time'] = 0
                if track['loss_time'] >= t_loss:
                    tracks_active.remove(track)
        else:
            if len(dets_2) > 0:
                best_match = max(dets_2, key=lambda x: iou(track['bboxes'], x['bbox']))
                if iou(track['bboxes'], best_match['bbox']) >= sigma_iou:
                    track['bboxes'] = best_match['bbox']
                    track['max_score'] = max(track['max_score'], best_match['score'])
                    track['life_time'] += 1
                    track['loss_time'] = 0
                    updated_tracks.append(track)
                    del dets_2[dets_2.index(best_match)]
                    if track['max_score'] >= sigma_h and track['life_time'] >= t_life:
                        if track['id'] == 0:
                            id += 1
                            track['id'] = id
                        tracks_finished.append(track)
            if len(updated_tracks) == 0 or track is not updated_tracks[-1]:
                track['loss_time'] += 1
                if track['loss_time'] > 5:
                    track['life_time'] = 0
                if track['loss_time'] >= t_loss:
                    tracks_active.remove(track)
    new_tracks_1 = [
        {'bboxes': det['bbox'], 'max_score': det['score'], 'life_time': 1, 'loss_time': 0, 'id': 0, 'cls': det['cls']}
        for det in dets_1]
    new_tracks_2 = [
        {'bboxes': det['bbox'], 'max_score': det['score'], 'life_time': 1, 'loss_time': 0, 'id': 0, 'cls': det['cls']}
        for det in dets_2]
    tracks_active = tracks_active + new_tracks_1 + new_tracks_2
    return tracks_finished, tracks_active, id


def convertBack(xmin, ymin, xmax, ymax, conf, cls):
    xmin = int(xmin)
    xmax = int(xmax)
    ymin = int(ymin)
    ymax = int(ymax)
    return [xmin, ymin, xmax, ymax], conf, int(cls)


def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = os.path.join(maindir, filename)
            image_names.append(apath)
    return image_names


def image_demo(input_path, output_path):
    device = select_device(0)
    model_l = DetectMultiBackend('baseline_yolov5l.engine', device=device, dnn=False, data='gaofen.yaml')
    stride = 32
    imgsz = 1920

    model_m = DetectMultiBackend('yolov5m.engine', device=device, dnn=False, data='gaofen.yaml')
    start_time = time.time()
    inference_times = []
    nms_times = []
    track_times = []
    for subdir in os.listdir(os.path.join(input_path)):
        tracks = {}
        tracks_active = []
        id = 0
        dataset = LoadImages(os.path.join(input_path, subdir, 'img'), img_size=imgsz, stride=stride)
        for path, img, im0s, vid_cap, _ in dataset:
            img = torch.from_numpy(img).to(device)
            img = img.float()
            img /= 255.0
            if len(img.shape) == 3:
                img = img[None]
            inference_start_time = time.time()
            pred_l = model_l(img, augment=False, visualize=False)
            pred_m = model_m(img, augment=False, visualize=False)
            inference_times.append(time.time() - inference_start_time)
            nms_start_time = time.time()
            pred_l = non_max_suppression(pred_l, conf_thres=0.2)
            pred_m = non_max_suppression(pred_m, conf_thres=0.2)
            nms_times.append(time.time() - nms_start_time)
            track_start_time = time.time()
            bboxes_l = []
            for i, det in enumerate(pred_l):
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for xmin, ymin, xmax, ymax, conf, cls in reversed(det):
                        bboxes_l.append([xmin.cpu().numpy(), ymin.cpu().numpy(), xmax.cpu().numpy(), ymax.cpu().numpy(),
                                         conf.cpu().numpy(), cls.cpu().numpy() + 1])

            bboxes_l = [convertBack(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]),
                                    float(bbox[5])) for bbox in bboxes_l]
            bboxes_m = []
            for i, det in enumerate(pred_m):
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)
                if len(det):
                    det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()
                    for xmin, ymin, xmax, ymax, conf, cls in reversed(det):
                        bboxes_m.append([xmin.cpu().numpy(), ymin.cpu().numpy(), xmax.cpu().numpy(), ymax.cpu().numpy(),
                                         conf.cpu().numpy(), cls.cpu().numpy() + 1])

            bboxes_m = [convertBack(float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3]), float(bbox[4]),
                                    float(bbox[5])) for bbox in bboxes_m]

            boxes_list = []
            boxes_list_l = []
            boxes_list_m = []
            scores_list = []
            scores_list_l = []
            scores_list_m = []
            labels_list = []
            labels_list_l = []
            labels_list_m = []
            weights = [1, 1]
            for bbox, conf, cls in bboxes_l:
                boxes_list_l.append([bbox[0] / 1920, bbox[1] / 1080, bbox[2] / 1920, bbox[3] / 1080])
                scores_list_l.append(conf)
                labels_list_l.append(cls)
            for bbox, conf, cls in bboxes_m:
                boxes_list_m.append([bbox[0] / 1920, bbox[1] / 1080, bbox[2] / 1920, bbox[3] / 1080])
                scores_list_m.append(conf)
                labels_list_m.append(cls)
            boxes_list.append(boxes_list_l)
            boxes_list.append(boxes_list_m)

            scores_list.append(scores_list_l)
            scores_list.append(scores_list_m)

            labels_list.append(labels_list_l)
            labels_list.append(labels_list_m)

            boxes, scores, labels = weighted_boxes_fusion(boxes_list, scores_list, labels_list, weights=weights,
                                                          iou_thr=0.35, skip_box_thr=0.2)


            bboxes = [convertBack(float(boxes[i][0]) * 1920 - 2, float(boxes[i][1]) * 1080 - 2, float(boxes[i][2]) * 1920 + 2, float(boxes[i][3]) * 1080 + 2, float(scores[i]),
                                  float(labels[i])) for i in range(len(boxes))]
            detections = [{'bbox': bbox, 'score': conf, 'cls': cls} for bbox, conf, cls in bboxes]
            tracks_finished, tracks_active, id = track_iou(detections, tracks_active, 0.05, 0.4, 0.01, 5, 10, id)
            for track_finished in tracks_finished:
                bbox = track_finished['bboxes']
                tracks.setdefault(track_finished['id'], []).append(
                    '{},{},{},{},{},{},{},{},{}'.format(int(path[-10:-4]), track_finished['id'], bbox[0] + 2,
                                                        bbox[1] + 2, bbox[2] - bbox[0] - 4, bbox[3] - bbox[1] - 4, 1, 1, 1))

                flag_1 = False
                flag_2 = False
                flag_3 = False
                flag_4 = False
                track = tracks.get(track_finished['id'])
                for trackitem in track[max(0, len(track) - 9):-1]:
                    if trackitem.startswith('{},{},'.format(int(path[-10:-4]) - 1, track_finished['id'])):
                        flag_1 = True
                    if trackitem.startswith('{},{},'.format(int(path[-10:-4]) - 2, track_finished['id'])):
                        flag_2 = True
                    if trackitem.startswith('{},{},'.format(int(path[-10:-4]) - 3, track_finished['id'])):
                        flag_3 = True
                    if trackitem.startswith('{},{},'.format(int(path[-10:-4]) - 4, track_finished['id'])):
                        flag_4 = True
                if not flag_1:
                    track.append('{},{},{},{},{},{},{},{},{}'.format(int(path[-10:-4]) - 1, track_finished['id'],
                                                                     bbox[0] + 2, bbox[1] + 2, bbox[2] - bbox[0] - 4,
                                                                     bbox[3] - bbox[1] - 4, 1, 1, 1))
                if not flag_2:
                    track.append('{},{},{},{},{},{},{},{},{}'.format(int(path[-10:-4]) - 2, track_finished['id'],
                                                                     bbox[0] + 2, bbox[1] + 2, bbox[2] - bbox[0] - 4,
                                                                     bbox[3] - bbox[1] - 4, 1, 1, 1))
                if not flag_3:
                    track.append('{},{},{},{},{},{},{},{},{}'.format(int(path[-10:-4]) - 3, track_finished['id'],
                                                                     bbox[0] + 2, bbox[1] + 2, bbox[2] - bbox[0] - 4,
                                                                     bbox[3] - bbox[1] - 4, 1, 1,
                                                                     1))
                if not flag_4:
                    track.append('{},{},{},{},{},{},{},{},{}'.format(int(path[-10:-4]) - 4, track_finished['id'],
                                                                     bbox[0] + 2, bbox[1] + 2, bbox[2] - bbox[0] - 4,
                                                                     bbox[3] - bbox[1] - 4, 1, 1, 1))
            track_times.append(time.time() - track_start_time)

        with open(os.path.join(output_path, subdir + '.txt'), 'w') as f:
            for trackvalue in tracks.values():
                trackvalue.sort(key=lambda trackitem: int(trackitem.split(',')[0]))
                for trackitem in trackvalue:
                    f.write(trackitem + '\n')
    print("time:{},inference_time:{},nms_time:{},track_time:{}".format(time.time() - start_time, sum(inference_times),
                                                                       sum(nms_times), sum(track_times)))


def main():
    image_demo('/input_path', '/output_path')


if __name__ == "__main__":
    main()
