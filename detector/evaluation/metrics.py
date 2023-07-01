import collections

def classification_metrics(gt_proba_info, gt_objs, thresh=0.5):
    count_map = collections.defaultdict(dict)  # class -> counts

    for gt, proba, class_name in gt_proba_info:
        if gt == 0 and proba >= thresh:
            count_map[class_name]['fp'] = count_map[class_name].get('fp', 0) + 1
        elif gt == 1 and proba >= thresh:
            count_map[class_name]['tp'] = count_map[class_name].get('tp', 0) + 1
        elif gt == 1 and proba < thresh:
            count_map[class_name]['fn_upper'] = count_map[class_name].get('fn_upper', 0) + 1

    for gt_obj in gt_objs:
        count_map[gt_obj.name]['p'] = count_map[gt_obj.name].get('p', 0) + 1

    return [{'name': name, **value} for name, value in count_map.items()]


def mean_average_precision(): pass