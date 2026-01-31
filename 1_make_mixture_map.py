import os,json,cv2,rawpy,math
import numpy as np
from tqdm import tqdm
import argparse
import sys

# 默认配置（如果未提供命令行参数）
CAMERA = "nikon"
DATA_ROOT = "/mnt/sda2/SMY/ccmnet/LSMI"
RAW = None  # 将在main函数中设置
VISUALIZE = False
ZERO_MASK = -1              # masking value for unresolved pixel where G = 0 in all image pairs
SAVE_SUBTRACTED_IMG = False # option for saving subtracted image (ex. _2, _3)

# 这些变量将在main函数中初始化
RAW_EXT = None
TEMPLETE = None
BLACK_LEVEL = None
BLACK_LEVEL_RAW = None
SATURATION = None
RAW_PATTERN = None

"""
cellchart contains 24 color patch coordinates (x, y)

0   1   2   3   4   5
6   7   8   9   10  11
12  13  14  15  16  17
18  19  20  21  22  23

Each color patch coordinates start from upper left, clockwise order
"""
CELLCHART = np.float32([[
    # Row 1
    [0.25, 0.25],   [2.75, 0.25],   [2.75, 2.75],   [0.25, 2.75],
    [3.00, 0.25],   [5.50, 0.25],   [5.50, 2.75],   [3.00, 2.75], 
    [5.75, 0.25],   [8.25, 0.25],   [8.25, 2.75],   [5.75, 2.75],
    [8.50, 0.25],   [11.00, 0.25],  [11.00, 2.75],  [8.50, 2.75],
    [11.25, 0.25],  [13.75, 0.25],  [13.75, 2.75],  [11.25, 2.75],
    [14.00, 0.25],  [16.50, 0.25],  [16.50, 2.75],  [14.00, 2.75],

    # Row 2  
    [0.25, 3.00],   [2.75, 3.00],   [2.75, 5.50],   [0.25, 5.50],
    [3.00, 3.00],   [5.50, 3.00],   [5.50, 5.50],   [3.00, 5.50],
    [5.75, 3.00],   [8.25, 3.00],   [8.25, 5.50],   [5.75, 5.50],
    [8.50, 3.00],   [11.00, 3.00],  [11.00, 5.50],  [8.50, 5.50],
    [11.25, 3.00],  [13.75, 3.00],  [13.75, 5.50],  [11.25, 5.50],
    [14.00, 3.00],  [16.50, 3.00],  [16.50, 5.50],  [14.00, 5.50],

    # Row 3
    [0.25, 5.75],   [2.75, 5.75],   [2.75, 8.25],   [0.25, 8.25],
    [3.00, 5.75],   [5.50, 5.75],   [5.50, 8.25],   [3.00, 8.25],
    [5.75, 5.75],   [8.25, 5.75],   [8.25, 8.25],   [5.75, 8.25],
    [8.50, 5.75],   [11.00, 5.75],  [11.00, 8.25],  [8.50, 8.25],
    [11.25, 5.75],  [13.75, 5.75],  [13.75, 8.25],  [11.25, 8.25],
    [14.00, 5.75],  [16.50, 5.75],  [16.50, 8.25],  [14.00, 8.25],

    # Row 4
    [0.25, 8.50],   [2.75, 8.50],   [2.75, 11.00],  [0.25, 11.00],
    [3.00, 8.50],   [5.50, 8.50],   [5.50, 11.00],  [3.00, 11.00],
    [5.75, 8.50],   [8.25, 8.50],   [8.25, 11.00],  [5.75, 11.00],
    [8.50, 8.50],   [11.00, 8.50],  [11.00, 11.00], [8.50, 11.00],
    [11.25, 8.50],  [13.75, 8.50],  [13.75, 11.00], [11.25, 11.00],
    [14.00, 8.50],  [16.50, 8.50],  [16.50, 11.00], [14.00, 11.00]
]])
MCCBOX = np.float32([[0.00, 0.00], [16.75, 0.00], [16.75, 11.25], [0.00, 11.25]])

def angular_distance(l1, l2):
    unit_l1 = l1 / np.linalg.norm(l1)
    unit_l2 = l2 / np.linalg.norm(l2)
    dot_product = np.dot(unit_l1, unit_l2)
    radian = np.arccos(dot_product)  # radian
    degree = math.degrees(radian)    # degree

    return degree

def make_grid(img1, img1_wb1, img12, img12_wb12, img12_wb1, img12_wb2, img2_wb2, rb_map):
    # convert RGB to BGR
    img1_wb1 = cv2.cvtColor(img1_wb1, cv2.COLOR_RGB2BGR)
    img2_wb2 = cv2.cvtColor(img2_wb2, cv2.COLOR_RGB2BGR)
    img12_wb12 = cv2.cvtColor(img12_wb12, cv2.COLOR_RGB2BGR)
    img12_wb1 = cv2.cvtColor(img12_wb1, cv2.COLOR_RGB2BGR)
    img12_wb2 = cv2.cvtColor(img12_wb2, cv2.COLOR_RGB2BGR)
    rb_map = cv2.cvtColor(rb_map, cv2.COLOR_RGB2BGR)
    
    # resize the side of the images 1/4 the length of the side
    img1 = cv2.resize(img1, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img1_wb1 = cv2.resize(img1_wb1, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12 = cv2.resize(img12, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12_wb12 = cv2.resize(img12_wb12, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12_wb1 = cv2.resize(img12_wb1, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img12_wb2 = cv2.resize(img12_wb2, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    img2_wb2 = cv2.resize(img2_wb2, dsize=(1000, 750), interpolation=cv2.INTER_AREA)
    rb_map = cv2.resize(rb_map, dsize=(1000, 750), interpolation=cv2.INTER_AREA)

    # make text image
    img1 = add_label(img1, "img1")
    img1_wb1 = add_label(img1_wb1, "img1_wb1")
    img12 = add_label(img12, "img12")
    img12_wb12 = add_label(img12_wb12, "img12_wb12")
    img12_wb1 = add_label(img12_wb1, "img12_wb1")
    img12_wb2 = add_label(img12_wb2, "img12_wb2")
    img2_wb2 = add_label(img2_wb2, "img2_wb2")
    rb_map = add_label(rb_map, "rb_map")
    
    # concatenate all
    col1 = np.hstack((img1, img1_wb1))
    col2 = np.hstack((img12, img12_wb12))
    col3 = np.hstack((img12_wb1, img12_wb2))
    col4 = np.hstack((img2_wb2, rb_map))
    return cv2.vconcat([col1, col2, col3, col4])

def add_label(img, name):
    """
    img                 : image matrix
    name                : string
    """
    text = np.zeros((100, img.shape[1], 3), np.uint8) + 255
    cv2.putText(text, name, (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 0), 2, cv2.LINE_AA)
    return np.vstack((img, text))
    
def get_rb_map(coefficient_map):
    h,w = coefficient_map.shape

    rb_map = np.zeros((h,w,3), dtype=np.uint8)
    rb_map[:,:,0] = coefficient_map * 255
    rb_map[:,:,2] = (1 - coefficient_map) * 255
    
    return rb_map

def apply_wb_raw(raw, illumination_map):
    """
    raw                 : rawpy.RawPy class
    illumination_map    : half size illumination map in RGB channel order
    """
    h,w,_ = illumination_map.shape
    rh,rw = raw.raw_image.shape
    margin_h = int((rh-h*2)/2)
    margin_w = int((rw-w*2)/2)

    raw_matrix = np.clip(raw.raw_image.copy().astype('int16') - BLACK_LEVEL_RAW,0,SATURATION)
    raw_multiplier = np.ones_like(raw.raw_image,dtype=np.float32)

    r_multiplier = illumination_map[:,:,1] / illumination_map[:,:,0]
    b_multiplier = illumination_map[:,:,1] / illumination_map[:,:,2]

    wb_matrix = np.tile(-RAW_PATTERN, (h,w)).astype(np.float32)
    wb_matrix[wb_matrix==0] = r_multiplier.reshape(-1)
    wb_matrix[wb_matrix==-1] = 1
    wb_matrix[wb_matrix==-2] = b_multiplier.reshape(-1)
    wb_matrix[wb_matrix==-3] = 1

    raw_multiplier[margin_h:margin_h+h*2,margin_w:margin_w+w*2] = wb_matrix
    raw_matrix_wb = raw_matrix * raw_multiplier + BLACK_LEVEL_RAW

    for i in range(rh):
        for j in range(rw):
            raw.raw_image[i,j] = raw_matrix_wb[i,j]

def get_patch_chroma(chroma_map, method, normalize='green'):
    """
    chroma_map  : (3, 6, 3) shape array (MCC, graypatch, chroma)
    
    method      : mean or max
                  mean - average three brightest patches from three MCCs excluding saturation
                  max  - only one brightest patch excluding saturation

    normalize   : 'green' or 'sum'
                  green - normalize chroma vector to G=1
                  sum   - normalize chroma vector to sum = 1

    returns     : maxChartIdx, maxPatchIdx, normalized rg-chroma
                  (-1, -1, chroma) for method = "mean"
    """
    assert chroma_map.shape == (3,6,3)
    
    maxChartIdx = -1
    maxPatchIdx = -1
    retChroma = np.array([0, 0, 0])

    for chartIdx in range(3):
        for patchIdx in range(6):
            chroma = chroma_map[chartIdx, patchIdx, :]
            if SATURATION in chroma:
                continue

            elif method == "max" and chroma[1] > retChroma[1]:
                maxChartIdx = chartIdx
                maxPatchIdx = patchIdx
                retChroma = chroma

            elif method == "mean":
                retChroma += chroma
                break
    # print(retChroma)
    rgChroma = retChroma / np.sum(retChroma)
    g1_chroma = retChroma / retChroma[1]

    # print(rgChroma, g1_chroma)

    if normalize == 'sum':
        return maxChartIdx, maxPatchIdx, list(rgChroma)
    elif normalize == 'green':
        return maxChartIdx, maxPatchIdx, list(g1_chroma)

def get_coefficient_map(img_1_wb, img_2_wb, zero_mask=-1):
    """
    zero_mask   : masking value for pixel where both G = 0
    returns     : img_1's illuminant coefficient r
    """

    denominator = img_1_wb[:,:,1] + img_2_wb[:,:,1]

    # compute coefficient. fill zero_mask value for invalid denominator (if G value from both image = 0)
    coefficient = img_1_wb[:,:,1] / np.clip(denominator, 0.0001, SATURATION)
    coefficient = np.where(denominator==0, zero_mask, coefficient)

    return coefficient

def get_illuminant_chroma(img, mcc_list):
    """
    img         : BGR image
    mcc_list    : MCC chart coordinate list

    returns     : numpy array with shape (3,6,3)
                  (MCC chart, patch, RGB channel sum)
    """
    chroma = np.zeros((3,6,3), dtype=int)

    for mcc_idx in range(len(mcc_list)):
        mcc = mcc_list[mcc_idx]
        src = MCCBOX
        dst = mcc

        # Get perspective transform matrix, apply transform to cellchart
        M = cv2.getPerspectiveTransform(src, dst)
        cellchart = cv2.perspectiveTransform(CELLCHART, M)
        cellchart = np.reshape(cellchart, (24,4,2))

        # Reduce the box size by 50%
        for i in range(24):
            centerPoint = np.sum(cellchart[i], axis=0) / 4
            for j in range(4):
                cellchart[i][j] = (cellchart[i][j] + centerPoint) / 2

        # generate mask for gray patches (18~23) & record chromaticity
        for i in range(18,24):
            mask = np.zeros_like(img)
            cell = np.array([[cellchart[i,0]], [cellchart[i,1]], [cellchart[i,2]], [cellchart[i,3]]]).astype(int)
            cv2.drawContours(mask, [cell], 0, (1,1,1), -1) # fill inside the contour
            maskedImage = img*mask

            # RGB channelwise sum & flip (BGR to RGB)
            sumRGB = np.flip(np.sum(maskedImage, axis=(0,1)))
            chroma[mcc_idx, i-18, :] = sumRGB

            if True in (maskedImage >= (SATURATION - BLACK_LEVEL)):
                chroma[mcc_idx, i-18, :] = SATURATION

    return chroma

def get_illumination_map(place, placeInfo):
    # directory configuration
    # TIFF文件路径：从dataset目录读取（0_cvt2tiff.py的输出）
    camera_tiff_paths = {
        'sony': '/mnt/sda2/SMY/dataset/sony',
        'galaxy': '/mnt/sda2/SMY/dataset/galaxy',
        'nikon': '/mnt/sda2/SMY/dataset/nikon'
    }
    tiff_base_dir = camera_tiff_paths.get(CAMERA)
    if not tiff_base_dir:
        print(f"错误: 未知相机 {CAMERA}")
        return placeInfo
    
    # TIFF文件输入路径（从dataset目录读取）
    src_path = os.path.join(tiff_base_dir, place) + "/"
    if not os.path.exists(src_path):
        print(f"错误: TIFF文件目录不存在: {src_path}")
        return placeInfo
    
    # Mixture Map输出路径（保存到data_root目录）
    # 选项1：保存到 data_root/{camera}/PlaceXXX/（需要创建PlaceXXX目录）
    # 选项2：保存到 data_root/{camera}/npy/（统一npy文件夹）
    # 使用选项1，保持与原始结构一致
    output_base_dir = os.path.join(DATA_ROOT, CAMERA)
    output_place_dir = os.path.join(output_base_dir, place)
    
    # 如果输出目录不存在，创建它
    if not os.path.exists(output_place_dir):
        os.makedirs(output_place_dir, exist_ok=True)
    
    # 保存路径（用于np.save）
    output_path = output_place_dir + "/"
    
    vis_path = os.path.join(DATA_ROOT, CAMERA+"_visualize")

    if os.path.isdir(vis_path) == False and VISUALIZE:
        os.makedirs(vis_path)

    # read place annotation data from json data
    numOfLights = placeInfo["NumOfLights"]
    mccScale = 2 #placeInfo["MCCScale"]
    mcc1 = placeInfo["MCCCoord"]["mcc1"]
    mcc2 = placeInfo["MCCCoord"]["mcc2"]
    mcc3 = placeInfo["MCCCoord"]["mcc3"]
    mcc_list = np.float32([mcc1, mcc2, mcc3]) / mccScale

    # make illumination map for single light source (1)
    if placeInfo["NumOfLights"] == 1:
        singleimage = place + "_1"
        tiff_path = src_path + singleimage + ".tiff"
        print(f"[单光源] 处理 {place}, NumOfLights=1, 读取TIFF: {tiff_path}")
        
        # 读取单光源图像
        img_1 = cv2.imread(tiff_path, cv2.IMREAD_UNCHANGED)
        if img_1 is None or img_1.size == 0:
            print(f"错误: 无法读取图像 {tiff_path}")
            print(f"      请确保TIFF文件存在: {tiff_path}")
            return placeInfo
        
        print(f"[单光源] 成功读取图像，尺寸: {img_1.shape}")
        img_1 = img_1.astype("int16")
        
        img_1 = np.clip(img_1 - BLACK_LEVEL, 0, SATURATION - BLACK_LEVEL)
        
        # 计算光源色度
        chroma_1 = get_illuminant_chroma(img_1, mcc_list)
        maxChart1, maxPatch1, placeInfo["Light1"] = get_patch_chroma(chroma_1, method="max", normalize='green')
        
        # 对于单光源场景，mixture map是全1（100%单一光源）
        # 生成全1的mixture map，shape为(H, W, 1)
        h, w = img_1.shape[:2]
        coefficient_1 = np.ones((h, w), dtype=np.float32)
        coefficient_map = coefficient_1[:, :, np.newaxis]  # shape: (H, W, 1)
        
        # 保存mixture map
        output_file = output_path + singleimage + ".npy"
        np.save(output_file, coefficient_map)
        print(f"[单光源] ✓ 已生成mixture map: {output_file}")
        print(f"       文件大小: {os.path.getsize(output_file) / 1024:.2f} KB")
        
        # 单光源场景没有混合比例统计
        placeInfo["CoeffVariance"] = 0.0
        placeInfo["CoeffSTD"] = 0.0
        
        if VISUALIZE:
            # 可视化单光源场景
            try:
                raw_1 = rawpy.imread(src_path + singleimage + RAW_EXT)
                img_1_wb1 = raw_1.postprocess(user_black=BLACK_LEVEL, user_wb=[placeInfo["Light1"][1]/placeInfo["Light1"][0], 1.0, placeInfo["Light1"][1]/placeInfo["Light1"][2], 1.0], no_auto_bright=True, half_size=True)
                img_1_wb1 = cv2.cvtColor(img_1_wb1, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(vis_path, place+"_IMG1_WB1.jpg"), img_1_wb1)
            except Exception as e:
                print(f"警告: 无法生成可视化图像: {e}")
        
        return placeInfo
    
    # 跳过多光源场景（NumOfLights == 2 或 3）
    elif placeInfo["NumOfLights"] == 2:
        print(f"跳过双光源场景: {place} (NumOfLights=2)")
        return placeInfo
    
    elif placeInfo["NumOfLights"] == 3:
        print(f"跳过三光源场景: {place} (NumOfLights=3)")
        return placeInfo
    
    else:
        print(f"警告: 未知的光源数量: {placeInfo.get('NumOfLights', 'N/A')} (Place: {place})")
        return placeInfo
    
    # 以下代码已禁用（只处理单光源场景）
    # 如果需要恢复多光源处理，请取消注释并删除上面的return语句
    if False:  # 永远不会执行
        # make illumination map of 2 images (1,12)
        if placeInfo["NumOfLights"] == 2:
            singleimage = place + "_1"
            multiimage = place + "_12"

        # prevent uint16 type subtraction underflow
        img_1 = cv2.imread(src_path + singleimage + ".tiff", cv2.IMREAD_UNCHANGED).astype("int16")
        img_12 = cv2.imread(src_path + multiimage + ".tiff", cv2.IMREAD_UNCHANGED).astype("int16")
        
        img_2 = np.clip(img_12 - img_1, 0, SATURATION - BLACK_LEVEL)
        img_1 = np.clip(img_1 - BLACK_LEVEL, 0, SATURATION - BLACK_LEVEL)

        if SAVE_SUBTRACTED_IMG:
            cv2.imwrite(src_path + place + "_2.tiff", img_2.astype("uint16"))

        # calculate MCC gray cellchart RGB value (shape 3,6,3)
        chroma_1 = get_illuminant_chroma(img_1, mcc_list)
        chroma_2 = get_illuminant_chroma(img_2, mcc_list)

        # get maximum chart,patch,chromaticity without saturation
        # json value is directly updated with calculated chromaticity
        maxChart1, maxPatch1, placeInfo["Light1"] = get_patch_chroma(chroma_1, method="max", normalize='green')
        maxChart2, maxPatch2, placeInfo["Light2"] = get_patch_chroma(chroma_2, method="max", normalize='green')

        light_1 = placeInfo["Light1"]
        light_2 = placeInfo["Light2"]

        # calculate angular distance between light 1 & 2
        placeInfo["AD"] = angular_distance(placeInfo["Light1"], placeInfo["Light2"])

        # calculate brightness (sum of G pixels) difference of light2-affected MCC
        before = np.sum(chroma_1[maxChart2, maxPatch2, 1])
        diff = np.sum(chroma_2[maxChart2, maxPatch2, 1])
        ratio = (diff / before) * 100.
        placeInfo["BrightnessDiff"] = ratio

        """
        From here, calculate each pixel's light combination coefficient map
        and save them as numpy array (.npy)
        """
        # generate coefficient (mixture) map from G channel
        coefficient_1 = get_coefficient_map(img_1, img_2, ZERO_MASK)
        coefficient_2 = np.where(coefficient_1==ZERO_MASK,ZERO_MASK,1.0 - coefficient_1)

        # save coefficient map
        coefficient_map = np.stack((coefficient_1, coefficient_2), axis=-1)
        # 保存到data_root目录（不是dataset目录）
        np.save(output_path + multiimage, coefficient_map)

        # calculate coefficient statistics
        masked_coefficient_map = coefficient_1[coefficient_1>-1]
        var_1 = np.var(masked_coefficient_map)
        var_2 = np.var(1-masked_coefficient_map)
        std_1 = np.std(masked_coefficient_map)
        std_2 = np.std(1-masked_coefficient_map)
        placeInfo["CoeffVariance"] = (var_1 + var_2) / 2
        placeInfo["CoeffSTD"] = (std_1 + std_2) / 2

        """
        ##########################################################################
        #  JPG Visualization - using RawPy                                       #
        #  If you use RawPy, lots of postprocess operations are performed.       #
        #  (auto brightness, 8bit sRGB colorspace transform, etc...)             #
        #  You can control them, by using arguments of postprocess() function.   #
        ##########################################################################
        """
        if VISUALIZE:
            # open two raw images (1,12)
            raw_1 = rawpy.imread(src_path + singleimage + RAW_EXT)
            raw_12 = rawpy.imread(src_path + multiimage + RAW_EXT)
            
            # subtract two raw images (12 - 1)
            raw_2 = rawpy.imread(src_path + multiimage + RAW_EXT)
            raw_12_matrix = raw_2.raw_image.copy().astype('int16')
            raw_1_matrix = raw_1.raw_image.copy().astype('int16')
            raw_2_matrix = np.clip(raw_12_matrix - raw_1_matrix, 0, SATURATION) + BLACK_LEVEL
            height, width = raw_2.sizes.raw_height, raw_2.sizes.raw_width
            for h in range(height):
                for w in range(width):
                    raw_2.raw_image[h,w] = raw_2_matrix[h,w]
            
            # compute mixed illumination map and apply WB
            illumination_map_12 = np.stack((coefficient_1,)*3, axis=2) * [[light_1]] \
                                + np.stack((coefficient_2,)*3, axis=2) * [[light_2]]
            z, y, x = np.where(coefficient_map == -1)
            for i in range(len(x)):
                illumination_map_12[z[i], y[i], x[i]] = 1/3
            
            apply_wb_raw(raw_12, illumination_map_12)
            img12_wb12 = raw_12.postprocess(user_black=BLACK_LEVEL, user_wb=[1,1,1,1], no_auto_bright=True, half_size=True)

            raw_12 = rawpy.imread(src_path + multiimage + RAW_EXT)
            rgb_12_awb = raw_12.postprocess(use_auto_wb=True, no_auto_bright=True, half_size=True)
            rgb_12_daylight = raw_12.postprocess(user_wb=raw_12.daylight_whitebalance, no_auto_bright=True, half_size=True)
            rgb_12_camera = raw_12.postprocess(user_wb=raw_12.camera_whitebalance, no_auto_bright=True, half_size=True)

            cv2.imwrite(os.path.join(vis_path, place+"_awb.png"), cv2.cvtColor(rgb_12_awb, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_path, place+"_daylight.png"), cv2.cvtColor(rgb_12_daylight, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_path, place+"_camera.png"), cv2.cvtColor(rgb_12_camera, cv2.COLOR_RGB2BGR))
            cv2.imwrite(os.path.join(vis_path, place+"_wb12.png"), cv2.cvtColor(img12_wb12, cv2.COLOR_RGB2BGR))

            # rb-map image
            rb_map = get_rb_map(coefficient_1)
            cv2.imwrite(os.path.join(vis_path, place+"_rbmap.jpg"), cv2.cvtColor(rb_map, cv2.COLOR_RGB2BGR))


    # make illumination map (1,12,13,123 pair)
    elif placeInfo["NumOfLights"] == 3:
        img_1_name = place + "_1"
        img_12_name = place + "_12"
        img_13_name = place + "_13"
        img_123_name = place + "_123"

        img_1 = cv2.imread(src_path + img_1_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        img_12 = cv2.imread(src_path + img_12_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        img_13 = cv2.imread(src_path + img_13_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        img_123 = cv2.imread(src_path + img_123_name + ".tiff", cv2.IMREAD_UNCHANGED) - BLACK_LEVEL
        
        # prevent uint16 type subtraction underflow
        img_1_int16 = img_1.astype("int16")
        img_12_int16 = img_12.astype("int16")
        img_13_int16 = img_13.astype("int16")
        img_123_int16 = img_123.astype("int16")

        # pixel level image subtraction
        img_2 = np.clip(img_12_int16 - img_1_int16, 0, SATURATION - BLACK_LEVEL)
        img_3 = np.clip(img_13_int16 - img_1_int16, 0, SATURATION - BLACK_LEVEL)
        img_23 = np.clip(img_123_int16 - img_1_int16, 0, SATURATION - BLACK_LEVEL)

        if SAVE_SUBTRACTED_IMG:
            cv2.imwrite(src_path + place + "_2.tiff", img_2.astype("uint16"))
            cv2.imwrite(src_path + place + "_3.tiff", img_3.astype("uint16"))
            cv2.imwrite(src_path + place + "_23.tiff", img_23.astype("uint16"))

        # calculate MCC gray cellchart RGB value (shape 3,6,3)
        chroma_1 = get_illuminant_chroma(img_1, mcc_list)
        chroma_2 = get_illuminant_chroma(img_2, mcc_list)
        chroma_3 = get_illuminant_chroma(img_3, mcc_list)

        # get maximum chart,patch,chromaticity without saturation
        # json value is directly updated with calculated chromaticity
        maxChart1, maxPatch1, placeInfo["Light1"] = get_patch_chroma(chroma_1, method="max", normalize='green')
        maxChart2, maxPatch2, placeInfo["Light2"] = get_patch_chroma(chroma_2, method="max", normalize='green')
        maxChart3, maxPatch3, placeInfo["Light3"] = get_patch_chroma(chroma_3, method="max", normalize='green')

        light_1 = placeInfo["Light1"]
        light_2 = placeInfo["Light2"]
        light_3 = placeInfo["Light3"]

        # calculate angular distance between lights
        placeInfo["AD12"] = angular_distance(placeInfo["Light1"], placeInfo["Light2"])
        placeInfo["AD23"] = angular_distance(placeInfo["Light2"], placeInfo["Light3"])
        placeInfo["AD31"] = angular_distance(placeInfo["Light3"], placeInfo["Light1"])

        """
        From here, calculate each pixel's light combination coefficient map
        and save them as 2-channel numpy array (.npy)
        """
        # generate coefficient map from G channel
        # cannot use get_coefficient_map function in 3 lights case
        denominator_13 = img_1[:,:,1] + img_3[:,:,1]
        denominator_12 = img_1[:,:,1] + img_2[:,:,1]
        denominator_123 = img_1[:,:,1] + img_2[:,:,1] + img_3[:,:,1]

        # compute coefficient. -1 for invalid denominator_123 (if G value from both image = 0)
        coefficient_1 = img_1[:,:,1] / np.clip(denominator_12, 0.0001, SATURATION)
        coefficient_1 = coefficient_1.clip(0, 1)
        coefficient_1 = np.where(denominator_12==0, ZERO_MASK, coefficient_1)
        coefficient_2 = img_2[:,:,1] / np.clip(denominator_12, 0.0001, SATURATION)
        coefficient_2 = coefficient_2.clip(0, 1)
        coefficient_2 = np.where(denominator_12==0, ZERO_MASK, coefficient_2)
        coefficient_map_12 = np.stack((coefficient_1, coefficient_2), axis=-1)
        np.save(output_path + img_12_name, coefficient_map_12)

        coefficient_1 = img_1[:,:,1] / np.clip(denominator_13, 0.0001, SATURATION)
        coefficient_1 = coefficient_1.clip(0, 1)
        coefficient_1 = np.where(denominator_13==0, ZERO_MASK, coefficient_1)
        coefficient_3 = img_3[:,:,1] / np.clip(denominator_13, 0.0001, SATURATION)
        coefficient_3 = coefficient_3.clip(0, 1)
        coefficient_3 = np.where(denominator_13==0, ZERO_MASK, coefficient_3)
        coefficient_map_13 = np.stack((coefficient_1, coefficient_3), axis=-1)
        np.save(output_path + img_13_name, coefficient_map_13)

        coefficient_2 = img_2[:,:,1] / np.clip(denominator_123, 0.0001, SATURATION)
        coefficient_2 = coefficient_2.clip(0, 1)
        coefficient_2 = np.where(denominator_123==0, ZERO_MASK, coefficient_2)
        coefficient_3 = img_3[:,:,1] / np.clip(denominator_123, 0.0001, SATURATION)
        coefficient_3 = coefficient_3.clip(0, 1)
        coefficient_3 = np.where(denominator_123==0, ZERO_MASK, coefficient_3)
        coefficient_1 = np.clip(1 - coefficient_2 - coefficient_3, 0, 1)
        coefficient_1 = np.where(denominator_123==0, ZERO_MASK, coefficient_1)
        coefficient_map = np.stack((coefficient_1, coefficient_2, coefficient_3), axis=-1)
        np.save(output_path + img_123_name, coefficient_map)

        # save coefficient statistics
        masked_coefficient_1 = coefficient_1[coefficient_1>-1]
        masked_coefficient_2 = coefficient_2[coefficient_2>-1]
        masked_coefficient_3 = coefficient_3[coefficient_3>-1]

        var_1 = np.var(masked_coefficient_1)
        var_2 = np.var(masked_coefficient_2)
        var_3 = np.var(masked_coefficient_3)
        std_1 = np.std(masked_coefficient_1)
        std_2 = np.std(masked_coefficient_2)
        std_3 = np.std(masked_coefficient_3)

        placeInfo["CoeffVariance"] = (var_1 + var_2 + var_3) / 3
        placeInfo["CoeffSTD"] = (std_1 + std_2 + std_3) / 3

        """
        ##########################################################################
        #  JPG Visualization - using RawPy                                       #
        #  If you use RawPy, lots of postprocess operations are performed.       #
        #  (auto brightness, 8bit sRGB colorspace transform, etc...)             #
        #  You can control them, by using arguments of postprocess() function.   #
        ##########################################################################
        """
        if VISUALIZE:
            # open two raw images (1,12, 13, 123)
            raw_1 = rawpy.imread(src_path + img_1_name + RAW_EXT)
            raw_12 = rawpy.imread(src_path + img_12_name + RAW_EXT)
            raw_13 = rawpy.imread(src_path + img_13_name + RAW_EXT)
            raw_123 = rawpy.imread(src_path + img_123_name + RAW_EXT)
            
            # subtract two raw images (12 - 1)
            raw_2 = rawpy.imread(src_path + img_12_name + RAW_EXT)
            raw_2_matrix = raw_2.raw_image
            raw_1_matrix = raw_1.raw_image
            height, width = raw_2.sizes.raw_height, raw_2.sizes.raw_width
            for h in range(height):
                for w in range(width):
                    if raw_2_matrix[h,w] < raw_1_matrix[h,w]:
                        raw_2_matrix[h,w] = 0
                    else:
                        raw_2_matrix[h,w] = raw_2_matrix[h,w] - raw_1_matrix[h,w]

            # subtract two raw images (123 - 12)
            raw_3 = rawpy.imread(src_path + img_123_name + RAW_EXT)
            raw_3_matrix = raw_3.raw_image
            raw_12_matrix = raw_12.raw_image
            height, width = raw_3.sizes.raw_height, raw_3.sizes.raw_width
            for h in range(height):
                for w in range(width):
                    if raw_3_matrix[h,w] < raw_12_matrix[h,w]:
                        raw_3_matrix[h,w] = 0
                    else:
                        raw_3_matrix[h,w] = raw_3_matrix[h,w] - raw_12_matrix[h,w]
            
            # compute mixed illumination map and apply WB
            illumination_map_123 = np.stack((coefficient_map[:,:,0],)*3, axis=2) * [[light_1]] \
                                + np.stack((coefficient_map[:,:,1],)*3, axis=2) * [[light_2]] \
                                + np.stack((coefficient_map[:,:,2],)*3, axis=2) * [[light_3]]
            z, y, x = np.where(coefficient_map == -1)
            for i in range(len(x)):
                illumination_map_123[z[i], y[i], x[i]] = 1/3

            apply_wb_raw(raw_123, illumination_map_123)

            # apply white balance & decode raw file to 3 channel image
            # img1_wb1 = raw_1.postprocess(user_wb=[light_1[1]/light_1[0], 1.0, light_1[1]/light_1[2], 1.0], no_auto_bright=True, half_size=True)
            # img2_wb2 = raw_2.postprocess(user_wb=[light_2[1]/light_2[0], 1.0, light_2[1]/light_2[2], 1.0], no_auto_bright=True, half_size=True)
            # img12_wb12 = raw_12.postprocess(user_wb=[1,1,1,1], no_auto_bright=True, half_size=True)

            img123_wb123 = raw_123.postprocess(user_black=BLACK_LEVEL, user_wb=[1,1,1,1], no_auto_bright=True, half_size=True)
            img123_wb123 = cv2.cvtColor(img123_wb123, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(vis_path, place+"_IMG123_WB123.jpg"), img123_wb123)

            # rb-map image
            rgb_map = np.zeros_like(coefficient_map, dtype=np.uint8)
            rgb_map[:,:,0] = coefficient_map[:,:,0] * 255
            rgb_map[:,:,1] = coefficient_map[:,:,1] * 255
            rgb_map[:,:,2] = coefficient_map[:,:,2] * 255
            cv2.imwrite(os.path.join(vis_path, place+"_coefficient_map.png"), cv2.cvtColor(rgb_map, cv2.COLOR_RGB2BGR))

            # calculate coefficient variance
            placeInfo["CoeffVariance"] = np.mean(np.var(coefficient_map, axis=(0,1)))

    # return Json data contains Light_RGB
    return placeInfo

if __name__ == "__main__":
    # 添加命令行参数支持
    parser = argparse.ArgumentParser(description="生成LSMI mixture map")
    parser.add_argument("--camera", type=str, default="galaxy",
                       choices=["galaxy", "sony", "nikon"],
                       help="相机名称 (default: galaxy)")
    parser.add_argument("--data_root", type=str, 
                       default="/mnt/sda2/SMY/ccmnet/LSMI",
                       help="LSMI数据根目录 (default: /mnt/sda2/SMY/ccmnet/LSMI)")
    parser.add_argument("--raw_file", type=str, default=None,
                       help="RAW模板文件路径（用于提取相机参数如黑电平、白电平，只需要一个示例ARW文件即可，如果未指定则自动查找）")
    parser.add_argument("--meta_json_path", type=str, default=None,
                       help="meta.json文件路径（如果未指定，根据相机自动确定）")
    parser.add_argument("--update_meta_json", action="store_true", default=True,
                       help="是否更新meta.json文件（默认True）")
    parser.add_argument("--output_meta_json_path", type=str, default=None,
                       help="更新后的meta.json保存路径（如果未指定，保存到原始路径）")
    parser.add_argument("--visualize", action="store_true", default=False,
                       help="生成可视化图像")
    parser.add_argument("--save_subtracted", action="store_true", default=False,
                       help="保存差分图像（_2, _3等）")
    
    args = parser.parse_args()
    
    # 设置全局变量
    CAMERA = args.camera
    DATA_ROOT = args.data_root
    VISUALIZE = args.visualize
    SAVE_SUBTRACTED_IMG = args.save_subtracted
    
    # 查找RAW模板文件
    if args.raw_file:
        RAW = args.raw_file
    else:
        # 自动查找RAW文件
        raw_extensions = {
            'sony': '.arw',
            'nikon': '.nef',
            'galaxy': '.dng'
        }
        raw_ext = raw_extensions.get(CAMERA, '.dng')
        
        # RAW文件路径（与TIFF文件在同一目录）
        camera_raw_paths = {
            'sony': '/mnt/sda2/SMY/dataset/sony',
            'galaxy': '/mnt/sda2/SMY/dataset/galaxy',
            'nikon': '/mnt/sda1/SMY/project/dataset/LSMI/nikon'
        }
        raw_base_dir = camera_raw_paths.get(CAMERA)
        
        # 尝试多个可能的路径
        raw_candidates = []
        
        # 1. 从dataset目录的Place目录中查找（与TIFF文件同目录）
        if raw_base_dir:
            print(f"调试: 查找RAW文件，基础目录: {raw_base_dir}")
            # 遍历所有Place目录，找到第一个包含RAW文件的目录
            if os.path.exists(raw_base_dir):
                places = [f for f in os.listdir(raw_base_dir) 
                         if os.path.isdir(os.path.join(raw_base_dir, f)) and f.startswith("Place")]
                # 对Place目录进行排序，优先选择数字较小的目录
                places.sort(key=lambda x: int(x.replace("Place", "")) if x.replace("Place", "").isdigit() else 999999)
                print(f"调试: 找到 {len(places)} 个Place目录，将按顺序查找")
                
                # 遍历Place目录，找到第一个包含RAW文件的目录
                found_raw = False
                for place_name in places:
                    place_dir = os.path.join(raw_base_dir, place_name)
                    if os.path.exists(place_dir):
                        all_files = os.listdir(place_dir)
                        # 不区分大小写查找RAW文件
                        raw_files = [f for f in all_files 
                                   if f.lower().endswith(raw_ext.lower())]
                        if raw_files:
                            print(f"调试: 在 {place_name} 目录中找到 {len(raw_files)} 个{raw_ext}文件")
                            print(f"调试: 使用RAW文件: {raw_files[0]}")
                            raw_candidates.append(os.path.join(place_dir, raw_files[0]))
                            found_raw = True
                            break
                
                if not found_raw:
                    print(f"警告: 遍历了所有Place目录，未找到任何{raw_ext}文件")
            else:
                print(f"调试: 路径不存在: {raw_base_dir}")
                # 尝试从data_root查找
                data_root_places = os.path.join(DATA_ROOT, CAMERA)
                if os.path.exists(data_root_places):
                    print(f"调试: 尝试从data_root查找: {data_root_places}")
                    places = [f for f in os.listdir(data_root_places) 
                             if os.path.isdir(os.path.join(data_root_places, f)) and f.startswith("Place")]
                    if places:
                        first_place = places[0]
                        place_dir = os.path.join(data_root_places, first_place)
                        if os.path.exists(place_dir):
                            all_files = os.listdir(place_dir)
                            raw_files = [f for f in all_files 
                                       if f.lower().endswith(raw_ext.lower())]
                            if raw_files:
                                raw_candidates.append(os.path.join(place_dir, raw_files[0]))
                                print(f"调试: 从data_root找到候选: {raw_candidates[-1]}")
        
        # 2. 回退路径（data_root目录）
        camera_dir = os.path.join(DATA_ROOT, CAMERA)
        raw_candidates.extend([
            os.path.join(camera_dir, "dng", f"{CAMERA}{raw_ext}"),
            os.path.join(camera_dir, f"{CAMERA}{raw_ext}"),
            os.path.join(DATA_ROOT, f"{CAMERA}{raw_ext}"),
            f"{CAMERA}{raw_ext}",  # 当前目录
        ])
        
        RAW = None
        for candidate in raw_candidates:
            if candidate and os.path.exists(candidate):
                RAW = candidate
                print(f"找到RAW模板文件: {RAW}")
                break
        
        if RAW is None:
            print(f"错误: 找不到RAW模板文件")
            print(f"请使用--raw_file指定RAW文件路径，或确保以下路径之一存在:")
            for candidate in raw_candidates:
                if candidate:
                    print(f"  - {candidate}")
            sys.exit(1)
    
    # 检查RAW文件是否存在
    if not os.path.exists(RAW):
        print(f"错误: RAW模板文件不存在: {RAW}")
        sys.exit(1)
    
    # 初始化RAW相关变量
    RAW_EXT = os.path.splitext(RAW)[1]
    try:
        TEMPLETE = rawpy.imread(RAW)
    except Exception as e:
        print(f"错误: 无法读取RAW文件 {RAW}: {e}")
        sys.exit(1)
    
    # 设置相机参数
    if CAMERA == 'sony':
        BLACK_LEVEL = 128
        BLACK_LEVEL_RAW = 512
        SATURATION = 4095
    else:
        BLACK_LEVEL = min(TEMPLETE.black_level_per_channel)
        BLACK_LEVEL_RAW = BLACK_LEVEL
        SATURATION = TEMPLETE.white_level
    RAW_PATTERN = TEMPLETE.raw_pattern.astype('int8')
    
    print(f"生成 {CAMERA} 相机的 mixture map...")
    print(f"数据根目录: {DATA_ROOT}")
    print(f"RAW模板文件: {RAW}")
    
    # 检查相机目录是否存在
    camera_dir = os.path.join(DATA_ROOT, CAMERA)
    if not os.path.exists(camera_dir):
        print(f"错误: 相机目录不存在: {camera_dir}")
        sys.exit(1)
    
    # 确定meta.json路径
    if args.meta_json_path:
        meta_path = args.meta_json_path
    else:
        # 根据相机自动确定路径
        camera_meta_paths = {
            'sony': '/mnt/sda2/SMY/dataset/sony/meta.json',
            'galaxy': '/mnt/sda2/SMY/dataset/galaxy/meta.json',
            'nikon': '/mnt/sda1/SMY/project/dataset/LSMI/nikon/meta.json'
        }
        meta_path = camera_meta_paths.get(CAMERA)
        if meta_path is None:
            # 回退到data_root下的路径
            meta_path = os.path.join(camera_dir, "meta.json")
    
    if not os.path.exists(meta_path):
        print(f"错误: meta.json不存在: {meta_path}")
        sys.exit(1)
    
    # 检查文件大小
    file_size = os.path.getsize(meta_path)
    if file_size == 0:
        print(f"警告: meta.json文件为空 (0 bytes): {meta_path}")
        print(f"将创建新的meta.json数据")
        jsonData = {}
    else:
        # open json annotation file
        try:
            with open(meta_path, 'r') as json_file:
                jsonData = json.load(json_file)
        except json.JSONDecodeError as e:
            print(f"错误: 无法解析meta.json文件: {e}")
            sys.exit(1)
    
    # 检查是否已包含Light字段
    has_light_fields = False
    if jsonData:
        first_place = next(iter(jsonData.values()))
        if isinstance(first_place, dict):
            has_light_fields = 'Light1' in first_place or 'Light2' in first_place or 'Light3' in first_place
    
    # 根据相机类型和meta.json状态决定是否更新
    # Sony: 如果meta.json为空或不包含Light字段，必须更新
    # Galaxy和Nikon: 如果meta.json已包含Light字段，不更新（只生成Mixture Map）
    should_update_meta = True  # 默认更新
    
    if has_light_fields:
        if CAMERA == 'sony':
            print(f"检测到Sony的meta.json已包含Light字段，将更新以确保准确性")
            should_update_meta = True
        else:
            print(f"检测到{CAMERA}的meta.json已包含Light字段，将保留现有Light值，只生成Mixture Map")
            should_update_meta = False
            # 如果用户明确指定了--update_meta_json，则尊重用户选择
            if args.update_meta_json:
                print(f"注意: 由于指定了--update_meta_json，将更新meta.json")
                should_update_meta = True
    else:
        print(f"meta.json不包含Light字段，将计算并更新")
        should_update_meta = True
    
    # 如果用户明确指定了--update_meta_json=False，则尊重用户选择
    if not args.update_meta_json:
        should_update_meta = False
        print(f"注意: 由于指定了--update_meta_json=False，将不更新meta.json")

    # initialize coefficient std array
    coeff_std = []
    coeff_var = []

    # get directory list (place目录)
    # 从meta.json中读取Place列表，而不是从文件系统查找
    # 因为Place目录在dataset目录下，不在data_root目录下
    if not jsonData:
        print(f"错误: meta.json为空，无法获取Place列表")
        sys.exit(1)
    
    places = sorted([place for place in jsonData.keys() if place.startswith("Place")])
    
    if not places:
        print(f"警告: meta.json中没有找到Place条目")
        print(f"请确保meta.json包含有效的Place数据")
    
    # 统计各类型场景数量
    single_light_count = 0
    multi_light_count = 0
    for place in places:
        if place in jsonData:
            num_lights = jsonData[place].get("NumOfLights", 0)
            if num_lights == 1:
                single_light_count += 1
            elif num_lights in [2, 3]:
                multi_light_count += 1
    
    print(f"\n场景统计:")
    print(f"  单光源场景 (NumOfLights=1): {single_light_count}")
    print(f"  多光源场景 (NumOfLights=2/3): {multi_light_count}")
    print(f"  总计: {len(places)}")
    print(f"\n开始处理...\n")

    for place in tqdm(places):
        # read json annotation
        if place not in jsonData:
            print(f"\n[跳过] {place}: 不在meta.json中（需要NumOfLights和MCCCoord信息）")
            continue
        
        placeInfo = jsonData[place].copy()  # 复制以避免修改原始数据
        
        # 打印当前处理的场景信息
        num_lights = placeInfo.get("NumOfLights", "N/A")
        if num_lights == 1:
            print(f"\n[处理中] {place}: NumOfLights={num_lights} (单光源)")
        elif num_lights == 2:
            print(f"[跳过] {place}: NumOfLights={num_lights} (双光源)")
        elif num_lights == 3:
            print(f"[跳过] {place}: NumOfLights={num_lights} (三光源)")
        else:
            print(f"[跳过] {place}: NumOfLights={num_lights} (未知类型)")
            continue  # 如果NumOfLights不是1/2/3，直接跳过
        
        # 检查是否已包含Light字段
        place_has_light = 'Light1' in placeInfo or 'Light2' in placeInfo or 'Light3' in placeInfo
        
        # 保存原有的Light值（如果存在）
        original_light_values = {}
        if place_has_light:
            if 'Light1' in placeInfo:
                original_light_values['Light1'] = placeInfo['Light1']
            if 'Light2' in placeInfo:
                original_light_values['Light2'] = placeInfo['Light2']
            if 'Light3' in placeInfo:
                original_light_values['Light3'] = placeInfo['Light3']
        
        # 调用get_illumination_map计算Mixture Map（总是计算，因为Mixture Map不依赖Light值）
        updated_placeInfo = get_illumination_map(place, placeInfo)
        
        # 如果原有Light值存在且meta.json已包含Light字段，恢复原有Light值
        # 这样可以避免重新计算Light值（如果TIFF文件未更新）
        if place_has_light and has_light_fields:
            for light_key, light_value in original_light_values.items():
                updated_placeInfo[light_key] = light_value
            # 也恢复其他可能被覆盖的字段
            if 'AD' in placeInfo:
                updated_placeInfo['AD'] = placeInfo['AD']
            if 'AD12' in placeInfo:
                updated_placeInfo['AD12'] = placeInfo['AD12']
            if 'AD23' in placeInfo:
                updated_placeInfo['AD23'] = placeInfo['AD23']
            if 'AD31' in placeInfo:
                updated_placeInfo['AD31'] = placeInfo['AD31']
            if 'BrightnessDiff' in placeInfo:
                updated_placeInfo['BrightnessDiff'] = placeInfo['BrightnessDiff']
        
        jsonData[place] = updated_placeInfo
        
        if "CoeffVariance" in jsonData[place] and "CoeffSTD" in jsonData[place]:
            coeff_var.append(jsonData[place]["CoeffVariance"])
            coeff_std.append(jsonData[place]["CoeffSTD"])

    if coeff_var and coeff_std:
        print("Coeff Variance :", np.mean(coeff_var))
        print("Coeff STD :", np.mean(coeff_std))
    else:
        print("警告: 没有计算到任何系数统计信息")

    # 保存json annotation（根据should_update_meta决定）
    if should_update_meta:
        output_path = args.output_meta_json_path if args.output_meta_json_path else meta_path
        try:
            with open(output_path, 'w') as out_file:
                json.dump(jsonData, out_file, indent=4)
            print(f"\n完成！已更新 meta.json: {output_path}")
        except Exception as e:
            print(f"错误: 无法保存meta.json到 {output_path}: {e}")
            sys.exit(1)
    else:
        print(f"\n完成！Mixture Map已生成，meta.json未更新")
        if not has_light_fields:
            print(f"警告: meta.json不包含Light字段，后续preprocess脚本可能失败")
            print(f"建议: 运行脚本时使用 --update_meta_json 来更新meta.json")
