import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os

# to calculate euclidian distance between 2 keypoints
def dist_keypoints(kp1,kp2):
    dist =  np.linalg.norm(np.array(kp1.pt) - np.array(kp2.pt))
    return dist


def perform(src ,dst):

    #creating SIFT object
    sift = cv.SIFT_create()
    kp1, des1 = sift.detectAndCompute(src, None)
    kp2, des2 = sift.detectAndCompute(dst, None)

    #calculating SIFT features for source and destination
    sift_img_s = cv.drawKeypoints(src, kp1, src, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    n_fea_s = len(kp1)
    sift_img_d = cv.drawKeypoints(dst, kp2, dst, (0, 255, 0), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    n_fea_d = len(kp2)

    #Brute force matcher to find the top 2 matches
    bf = cv.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    tot_matches = len(matches)

    #Lowe's ratio test to get good tests
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append([m])
    matches = good
    n_good = len(matches)
    matches = sorted(matches, key=lambda f: f[0].distance)

    #top20 scoring matches
    top20fig = cv.drawMatchesKnn(src, kp1, dst, kp2, matches[:20], None,
                                 flags=cv.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS, matchColor=(0, 255, 0))

    #Extracting points to perform homography
    src_pts = np.float32([kp1[m[0].queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m[0].trainIdx].pt for m in matches]).reshape(-1, 1, 2)

    M, mask = cv.findHomography(src_pts, dst_pts, cv.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    # Number of Inlier matches based on Mask
    n_inliers = sum(1 for i in mask if i == 1)

    #showing the boundaries of the object in the destination
    h, w = src.shape
    pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    dst_0 = cv.perspectiveTransform(pts, M)
    dst = cv.polylines(dst, [np.int32(dst_0)], True, 255, 30, cv.LINE_AA)

    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       flags=2)

    #Porojection of source keypoints in the destination image
    kp_proj = cv.perspectiveTransform(np.array([[[kp.pt[0], kp.pt[1]] for kp in kp1]]), M)[0]
    kp_proj = np.array([cv.KeyPoint(pt[0], pt[1], 1) for pt in kp_proj.tolist()])

    #Sorting the matches to find the top 10
    new_matches = sorted(matches, key=lambda match: dist_keypoints(kp_proj[match[0].queryIdx],kp2[match[0].trainIdx]))

    # show top 10 matches after homography transformation
    img3 = cv.drawMatchesKnn(src, kp1, dst, kp2,
                                             new_matches[:10],None, **draw_params)

    return sift_img_s, sift_img_d, n_fea_s, n_fea_d, top20fig, tot_matches,n_good, n_inliers, img3, M


def main():
    img_dir = './HW3_data'
    src = []
    dst = []
    for i in os.listdir(img_dir):
        if 'src' in i:
            src.append(i)
        if 'dst' in i:
            dst.append(i)

    for source in src:
        for destination in dst:
            src_img = cv.imread(img_dir+'/'+source,cv.IMREAD_GRAYSCALE)
            dst_img = cv.imread(img_dir+'/'+destination,cv.IMREAD_GRAYSCALE)
            sift_img_s, sift_img_d, n_fea_s, n_fea_d, top20_img, tot_matches,n_good, n_inliers, pers_img, M = perform(src_img,dst_img)
            cv.imwrite('SIFT_'+source,sift_img_s)
            cv.imwrite('SIFT_'+destination,sift_img_d)
            cv.imwrite('Top20'+source+'_'+destination,top20_img)
            cv.imwrite('Homo'+source+'_'+destination,pers_img)
            print('Number of SIFT features in '+source,n_fea_s)
            print('Number of SIFT features in '+destination,n_fea_d)
            print('Good matches found for pair'+source+' and '+destination+': ',n_good)
            print('Total number of matches',tot_matches)
            print('Number of inliers:',n_inliers)
            print('\nHomography matrix:\n\n',M)

if __name__ == "__main__":
    main()


