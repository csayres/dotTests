import fitsio
import matplotlib.pyplot as plt
import numpy
import skimage
from skimage.filters import sobel, gaussian
import glob
import PyGuide

from skimage.measure import regionprops, label
from sklearn.cluster import KMeans
from skimage.feature import blob_log, blob_dog, blob_doh
from skimage.transform import AffineTransform

from fibermeas.plotutils import plotCircle, imshow
from coordio.fitData import TransRotScaleModel, ModelFit, QuadrupoleModel
import time

_imgs = glob.glob("mediandots*.fits")
_imgs2 = glob.glob("meandots*.fits")

imgs = _imgs + _imgs2

rowPix = 3660
colPix = 5488

rowCen = int(rowPix/2)
colCen = int(colPix/2)

MaxCounts = 4095

# 0's indicate valid pixels
mask = numpy.ones((rowPix, colPix))
mask[:3020, 650:4585] = 0

# 1's indicate valid pixels
invmask = numpy.zeros((rowPix, colPix))
invmask[:3020, 650:4585] = 1

# https://www.edmundoptics.com/p/150-x-150mm-1000mm-spacing-opal-distortion-target/18703/
# 63-991
# 1 mm spacing
# 0.5 diameter
# .001 center to center
# .003 grid corner to grid corner
# grid size is 150 x 150

# camera BAS ACA5472-17UM
# pixel size = 2.4 micron
# lens FUJ HF3520-12M

# https://scikit-image.org/docs/dev/user_guide/numpy_images.html#coordinate-conventions

# SK mediandots_0.005.fits might be best
bestfile = "mediandots_0.005.fits"

# note that PosMinusIndex in pyGuide needs to be 0 to match
# skimage measurment conventions...

# http://wyant.optics.arizona.edu/zernikes/Zernikes.pdf

def getImgData(imgName):
    data = fitsio.read(imgName)
    data = skimage.util.invert(data)
    data = data - numpy.min(data)
    return data


def parseFile(filename):
    with open(filename, "r") as f:
        lines = f.readlines()

    arr = []
    for line in lines:
        row = [float(x) for x in line.split(",")]
        arr.append(row)

    return numpy.array(arr)
# imgs = ["mediandots_0.006.fits"]


def findWithPyGuide():
    CCDInfo = PyGuide.CCDInfo(
        bias=1,    # image bias, in ADU
        readNoise=1, # read noise, in e-
        ccdGain=1,  # inverse ccd gain, in e-/ADU
    )
    # this is junk but (0,0) is center of LL pixel
    PyGuide.Constants.PosMinusIndex = 0.0

    for img in imgs:
        data = getImgData(img)

        t1 = time.time()
        ctrDataList, imStats = PyGuide.findStars(
            data=data,
            mask=mask,
            satMask=None,
            ccdInfo=CCDInfo
        )
        print("found centroids", len(ctrDataList), "in", time.time()-t1)
        plt.figure(figsize=(8,8))
        plt.title(img)
        plt.imshow(data, origin="lower")
        xyErrs = []

        filename = "PyGuide_%s.txt"%img
        with open(filename, "w") as f:
            for ctrData in ctrDataList:
                xCtr, yCtr = ctrData.xyCtr
                rad = ctrData.rad
                xErr, yErr = ctrData.xyErr
                asymm = ctrData.asymm
                f.write("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n"%(xCtr,yCtr,xErr,yErr,rad,asymm))
                xyErrs.append(numpy.linalg.norm(ctrData.xyErr))
                plotCircle(ctrData.xyCtr[0], ctrData.xyCtr[1], ctrData.rad)
        print("%s mean error px"%img, numpy.mean(xyErrs))
    plt.show()


## skimage blob detection isn't as good as region..
# probably because dots are not gaussian
# def blobSkimage():
#     for img in imgs:
#         if img != bestfile:
#             continue
#         data = getImgData(img)

#         data = data * invmask
#         # thresh = (data > numpy.max(data) - 1000 ) # 1000 counts below max

#         data = data[rowCen-300:rowCen+300, colCen-300:colCen+300]
#         dataBlur = gaussian(data, sigma=4)

#         t1 = time.time()
#         minSigma = 4
#         maxSigma = 6
#         output = blob_dog(
#             dataBlur, min_sigma=minSigma, max_sigma=maxSigma, overlap=1
#         )

#         # output = blob_log(
#         #     data
#         # )

#         # throw out junk
#         # keep = output[:,2] < 10
#         # output = output[keep]

#         # plt.figure()
#         # plt.hist(output[:,2], bins=100)
#         # plt.show()

#         print("took", time.time()-t1)
#         plt.figure(figsize=(8,8))
#         imshow(data, doExtent=False)

#         # plot blob detector
#         # for row in output:
#         #     plotCircle(row[1], row[0], numpy.sqrt(2)*row[2])
#         # plt.show()

#         thresh = (data > numpy.max(data) - 1000 ) # 1000 counts below max
#         labels = label(thresh)
#         props = regionprops(labels, data)
#         for region in props:
#             yCent, xCent = region.weighted_centroid
#             rad = region.equivalent_diameter/2
#             plotCircle(xCent, yCent, rad*1.2, color="white")


#         PyGuide.Constants.PosMinusIndex = 0.0
#         CCDInfo = PyGuide.CCDInfo(
#             bias=1,    # image bias, in ADU
#             readNoise=1, # read noise, in e-
#             ccdGain=1,  # inverse ccd gain, in e-/ADU
#         )

#         ctrDataList, imStats = PyGuide.findStars(
#             data=data,
#             mask=None,
#             satMask=None,
#             ccdInfo=CCDInfo
#         )

#         for ctrData in ctrDataList:
#             xCtr, yCtr = ctrData.xyCtr
#             rad = ctrData.rad
#             plotCircle(xCtr, yCtr, rad*0.9, color="green")

#         plt.show()

        # import pdb; pdb.set_trace()


def centroidSkimage():
    # unlike pyguide, now 0,0 is corner of LL pixel
    for img in imgs:
        if img != bestfile:
            continue
        data = getImgData(img)
        print("on", img)

        # plt.figure()
        # imshow(data)
        # plt.show()

        data = data * invmask

        # calculate threshold using kmeans to find the two classes
        # (brightt and dark)
        # flatData = data.flatten()
        # kmeans = KMeans(n_clusters=2).fit(flatData)
        # print("cluster centers", kmeans.cluster_centers_)
        # import pdb; pdb.set_trace()
        # meanI = numpy.mean(flatData)


        # data = data[rowCen-150:rowCen+150, colCen-150:colCen+150] # center
        # data = data[2775:2775+150, 700:700+150] # top left corner
        # thresh = (data > numpy.percentile(data, 95))

        thresh = (data > numpy.max(data) - 1000 ) # 1000 counts below max

        # plt.figure()
        # imshow(thresh)
        # plt.show()

        t1 =  time.time()
        labels = label(thresh)
        print("%s labels took %.2f seconds" % (img, time.time()-t1))
        t1 = time.time()
        props = regionprops(labels, data)
        print("%s props took %.2f seconds" % (img, time.time()-t1))

        print("found %i props"%len(props))
        plt.figure(figsize=(8,8))
        plt.title(img)
        imshow(data)  # extent is set correctly so centers line up
        centErr = []
        ecen = []

        t1 = time.time()
        print(img)
        filename = "SK_%s.txt"%img
        with open(filename, "w") as f:
            for region in props:
                _yCent, _xCent = region.centroid
                yCent, xCent = region.weighted_centroid
                centErr.append(numpy.sqrt((yCent-_yCent)**2+(xCent-_xCent)**2))
                ecen.append(region.eccentricity)
                rad = region.equivalent_diameter/2
                f.write("%.5f, %.5f, %.5f, %.5f, %.5f, %.5f\n"%(xCent, yCent, _xCent, _yCent, region.eccentricity, rad))
                plotCircle(xCent, yCent, rad)
            print("plotting took", time.time()-t1)
            print("mean cent error", numpy.mean(centErr))
            print("mean eccentricity", numpy.mean(ecen))
            print("")
    plt.show()
        # break


def associate():
    topLX, topLY = 664.61489, 2980.8192  # top left dot
    r1 = 7.0223764

    x2, y2, = 4538.7292, 2994.2876 # top right dot
    r2 = 6.4945258
    # scale in mm/pixel
    roughScale = 149/numpy.linalg.norm([topLX-x2, topLY-y2])
    # angle
    _x = x2-topLX
    _y = y2-topLY
    roughAngle = numpy.arctan2(_y, _x)


    # middle of chip
    x1, y1 = 2453.7021, 1308.4415
    x2, y2 = 2608.5536, 1309.2168
    roughScale = 6/numpy.linalg.norm([x1-x2, y1-y2])


    rotMat = numpy.array([
        [numpy.cos(-roughAngle), -numpy.sin(-roughAngle)],
        [numpy.sin(-roughAngle), numpy.cos(-roughAngle)]
    ])

    for centType in ["SK"]:
        for img in imgs:
            if img != bestfile:
                continue
            print("")
            print("")
            print(centType, img)
            filename = "%s_%s.txt" % (centType, img)
            centData = parseFile(filename)[:, 2:4]  # just xy
            print(centData.shape)
            topLeft = numpy.array([[topLX,topLY]]*len(centData))
            print(topLeft.shape)
            diff = centData - topLeft
            dist = numpy.linalg.norm(diff, axis=1)
            arg = numpy.argmin(dist)
            if dist[arg] > 5: # within 5 pixels
                print("cannot find top left dot skipping")
                continue  # closer that 5 pixels

            # scale data to roughly mm
            centData = centData * roughScale

            # set origin to top left dot
            centData = centData - centData[arg,:]

            # rotate by rough angle
            centData = rotMat.dot(centData.T).T

            xyExpect = []
            xyMeas = []
            missingDots = 0
            for yDot in range(150):
                yDot = -1*yDot
                nXs = 0
                for xDot in range(150):
                    xyTest = numpy.array([xDot,yDot])
                    diff = centData - xyTest
                    dist = numpy.linalg.norm(diff, axis=1)
                    arg = numpy.argmin(dist)
                    if dist[arg] > 1:  # throw out measurements > 800 micron error, they're no good
                        # print("missed dot, continuing")
                        missingDots += 1
                        continue
                    nXs += 1
                    xyExpect.append(numpy.array([xDot, yDot]))
                    xyMeas.append(numpy.array(centData[arg,:]))
                if yDot == 0:
                    print("got %i xs"%nXs)

            xyMeas = numpy.array(xyMeas)
            xyExpect = numpy.array(xyExpect)

            print("found", len(xyMeas), "dots")
            print("missing", missingDots, "dots")

            dxy = xyMeas - xyExpect

            dx = dxy[:,0]
            dy = dxy[:,1]

            # plt.figure()
            # plt.hist(numpy.sqrt(dx**2+dy**2)*1000, bins=100)
            # plt.show()

            rms = numpy.sqrt(numpy.mean(dx**2+dy**2))
            print("rms micron", rms*1000)

            plt.figure(figsize=(9,9))
            plt.title(filename)
            plt.quiver(xyMeas[:,0], xyMeas[:,1], dx, dy, angles="xy", scale=20)
            plt.axis("equal")

            nPts = len(xyMeas)
            assocArray = numpy.zeros((nPts, 4))
            assocArray[:,0] = xyMeas[:,0]
            assocArray[:,1] = xyMeas[:,1]
            assocArray[:,2] = xyExpect[:,0]
            assocArray[:,3] = xyExpect[:,1]
            numpy.savetxt(filename+".npy", assocArray)
    # plt.show()

            # print(arg, dist[arg])

            # return


def fit():
    # https://scikit-image.org/docs/dev/auto_examples/transform/plot_matching.html

    filename = "SK_mediandots_0.005.fits.txt.npy"
    data = numpy.loadtxt(filename)
    xyMeas = data[:,:2]
    xyExpect = data[:,2:]
    rad = data[:,-1]
    plt.figure()
    plt.hist(rad, bins=200)
    plt.show()

    meanXY = numpy.mean(xyMeas, axis=0)
    xyMeas = xyMeas - meanXY
    xyExpect = xyExpect - meanXY

    # keep = numpy.abs(xyMeas[:,0]) < 30

    # xyMeas = xyMeas[keep]
    # xyExpect = xyExpect[keep]

    # keep = numpy.abs(xyMeas[:,1]) < 15
    # xyMeas = xyMeas[keep]
    # xyExpect = xyExpect[keep]

    print(xyMeas.shape)
    print(xyExpect.shape)

    # move origin to center of field
    # find center of camera divergene?
    # xyMeas + err = xyExpect


    # use the plot to find the center of divergence...
    err = xyExpect - xyMeas
    magErr = numpy.linalg.norm(err, axis=1)

    # throw out outliers
    keep = magErr*1000 < 190
    xyMeas = xyMeas[keep]
    print("threw out", len(xyExpect)-len(xyMeas))
    xyExpect = xyExpect[keep]
    err = err[keep]

    # plt.figure()
    # plt.hist(magErr*1000, bins=100)
    # plt.show()
    plt.figure(figsize=(9,9))
    plt.quiver(xyMeas[:,0], xyMeas[:,1], err[:,0], err[:,1], angles="xy", scale=1.5)
    plt.axis("equal")
    plt.show()

    # xMid = 81.5
    # yMid = -72.5

    # xyMeas = xyMeas - numpy.array([xMid, yMid])
    # xyExpect = xyExpect - numpy.array([xMid, yMid])

    print("RMS err prefit (micron)", numpy.sqrt(numpy.mean(err**2))*1000)


    model = AffineTransform()
    isOK = model.estimate(xyMeas, xyExpect)
    if not isOK:
        raise RuntimeError("affine fit failed")
    xyFit = model(xyMeas)


    err = xyExpect - xyFit
    print("RMS err postfit (micron)", numpy.sqrt(numpy.mean(err**2))*1000)
    # import pdb; pdb.set_trace()


    err = xyExpect -  xyFit
    plt.figure(figsize=(9,9))
    plt.quiver(xyFit[:,0], xyFit[:,1], err[:,0], err[:,1], angles="xy", scale=1.5)
    plt.axis("equal")
    plt.show()

    # trsModel = TransRotScaleModel()
    # trsModel = QuadrupoleModel()
    # trsFit = mf = ModelFit(
    #     model=trsModel,
    #     measPos=xyFit,
    #     nomPos=xyExpect,
    #     doRaise=True
    # )

    # # xyOff, rotAngle, scale = trsFit.model.getTransRotScale()
    # # print("xy translation (micron)", xyOff * 1000)
    # # print("rot (deg)", rotAngle)
    # # print("scale", scale)

    # posErr = trsFit.getPosError()
    # print("posErr shape", posErr.shape)
    # rmsErr = numpy.sqrt(numpy.mean(posErr[:,0]**2 + posErr[:,1]**2))*1000
    # print("fit rms error (micron)", rmsErr)


    # xyApply = trsModel.apply(xyFit, doInverse=True)
    # print("mean err posfit2 (micron)", numpy.mean(numpy.linalg.norm(xyApply-xyExpect, axis=1))*1000)

    # err = xyExpect - xyApply

    # plt.figure(figsize=(8,8))
    # plt.quiver(xyApply[:,0], xyApply[:,1], err[:,0], err[:,1], angles="xy", scale=1)
    # plt.axis("equal")
    # plt.show()



if __name__ == "__main__":
    # centroidSkimage()
    # associate()
    fit()






"""
Pyguide summary:
found centroids 17470 in 1115.1218650341034
mediandots_0.001.fits mean error px 0.5733844822716175
found centroids 12457 in 842.449245929718
mediandots_0.006.fits mean error px 1.042507885494916
found centroids 17238 in 1210.1206440925598
mediandots_0.010.fits mean error px 0.8101053860166332
found centroids 16301 in 1116.7551679611206
mediandots_0.008.fits mean error px 0.8783760734513005
found centroids 17428 in 1225.8801662921906
mediandots_0.005.fits mean error px 0.5534137333519673
found centroids 17467 in 1193.7263021469116
mediandots_0.003.fits mean error px 0.5645482298225228
found centroids 12389 in 864.9349727630615
meandots_0.006.fits mean error px 1.0139918157579073
found centroids 17472 in 1181.9330270290375
meandots_0.001.fits mean error px 0.5752984206437881
found centroids 17471 in 1217.650855064392
meandots_0.003.fits mean error px 0.5643244498392236
found centroids 16101 in 1088.2285590171814
meandots_0.008.fits mean error px 1.0251576179161526
found centroids 17125 in 1199.1868779659271
meandots_0.005.fits mean error px 0.599100617192079
"""




# data = fitsio.read("bcam1-0001.fits")
# data = skimage.util.invert(data)
# data = data - numpy.min(data)
# data[data < 2500] = 0
# print(numpy.min(data), numpy.max(data))
# # data = (data-MaxCounts)*-1
# plt.figure(figsize=(8,8))
# plt.imshow(data, origin="lower")
# plt.show()

