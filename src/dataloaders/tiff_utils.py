import rasterio
import rasterio as rio


def open_tiff(name_file):
    '''
    :param name_file: name of the tif file to open
    :return: rasterio.io.DatasetReader
    '''
    rasterio_img = rasterio.open(name_file)
    return rasterio_img

def read_tiff_to_np_array(tif_file):
    '''
    :param tif_file_for_single_date: tif file with labels for one date
    :return: converted numpy array of the tif image
    '''

    img = open_tiff(tif_file)
    img_array = img.read()[0]
    return img_array


def get_tiff_nb_bands(rasterio_img):
    '''
    get the number of bands (or layers) of a tif image
    :param rasterio_img: rasterio.io.DatasetReader or rasterio.io.DatasetWriter
    :return: int
    '''
    return rasterio_img.count

def get_tiff_band(rasterio_img, num_band):
    '''
    get one of the bands of the tif image as a numpy array
    :param rasterio_img: rasterio.io.DatasetReader or rasterio.io.DatasetWriter
    :param num_band: int from 1 to get_tif_nb_bands(rasterio_img)
    :return: np.array
    '''
    if 0 < num_band <= get_tiff_nb_bands(rasterio_img):
        mat = rasterio_img.read(num_band)
    else:
        print('Error. The band does not exist in this rasterio image.')
        raise IOError
    return mat

def get_tiff_transform(rasterio_img):
    '''
    get the transformation object of a tiff image (offset rotation...)
    :param rasterio_img: rasterio.io.DatasetReader or rasterio.io.DatasetWriter
    :return: transform affine.Affine (or else?)
    '''
    transform = rasterio_img.meta['transform']
    return transform


def get_tiff_crs(rasterio_img):
    '''
    get the crs object of a tiff image
    :param rasterio_img: rasterio.io.DatasetReader or rasterio.io.DatasetWriter
    :return:
    '''
    return rasterio_img.meta['crs']


def save_array_to_tiff(mat, filesaving, transform=rio.Affine(1, 0.0, 0.0, 0.0, 1, 0.0), crs='+proj=latlong'):
    '''
    save a numpy array to a rasterio tiff image
    :param mat: matrix array of size (template.height, template.width)
    :param transform: a rasterio transform object (affine.Affine for ex.) Use
    :param filesaving: path + name + '.tiff'
    :return: 0
    '''
    new_dataset = rasterio.open(filesaving, 'w', driver='GTiff', height=mat.shape[0],
                                width=mat.shape[1], count=1, dtype=mat.dtype,
                                crs=crs, transform=transform)
    new_dataset.write(mat, 1)
    new_dataset.close()

def save_multiple_arrays_to_tiff(mats, transform, filesaving):
    '''
    save a list of numpy arrays to a rasterio tiff image
    :param mats: list of matrix arrays of size (template.height, template.width)
    :param transform: a rasterio transform object (affine.Affine for ex.) Use
    :param filesaving: path + name + '.tiff'
    :return: 0
    '''
    new_dataset = rasterio.open(filesaving, 'w', driver='GTiff', height=mats[0].shape[0],
                                width=mats[0].shape[1], count=len(mats), dtype=mats[0].dtype,
                                crs='+proj=latlong', transform=transform)
    for i, mat in enumerate(mats):
        new_dataset.write_band(i+1, mat)
    new_dataset.close()