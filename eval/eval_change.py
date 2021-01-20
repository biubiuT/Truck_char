import json
from .script import getresult
#json输出转ICDAR15评价的输出模式
def order_points_new(pts):
    # sort the points based on their x-coordinates
    xSorted = pts[np.argsort(pts[:, 0]), :]

    # grab the left-most and right-most points from the sorted
    # x-roodinate points
    leftMost = xSorted[:2, :]
    rightMost = xSorted[2:, :]
    if leftMost[0,1]!=leftMost[1,1]:
        leftMost=leftMost[np.argsort(leftMost[:,1]),:]
    else:
        leftMost=leftMost[np.argsort(leftMost[:,0])[::-1],:]
    (tl, bl) = leftMost
    if rightMost[0,1]!=rightMost[1,1]:
        rightMost=rightMost[np.argsort(rightMost[:,1]),:]
    else:
        rightMost=rightMost[np.argsort(rightMost[:,0])[::-1],:]
    (tr,br)=rightMost
    # return the coordinates in top-left, top-right,
    # bottom-right, and bottom-left order
    return np.array([tl, tr, br, bl], dtype="int32")

json_path = 'C:/Users/WIN10/Desktop/test_paper.json'
file = open(json_path, "rb")
fileJson = json.load(file)
root_path = 'D:/Graduate/data/Graduate_v3/2021_re_data/AE_test_paper/'

for test_result in fileJson:
    name = test_result['img_name'].split('.')[0].split('_')[-1]
    save_path = root_path + 'res_img_' + str(int(name)) + '.txt'
    points = test_result['points']
    with open(save_path, 'w') as f:
        for i, box in enumerate(points):
            result = order_points_new(np.array(box))
            poly = np.array(result).reshape(-1)
            strResult = ','.join([str(p) for p in poly]) + '\r\n'
            f.write(strResult)

#生成评价指标
#if __name__ == '__main__':
    #getresult()