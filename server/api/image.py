#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cherrypy

from girder.api import access
from girder.api.rest import Resource, RestException, loadmodel
from girder.api.describe import Description, describeRoute
from girder.constants import AccessType
from girder.models.model_base import GirderException


class ImageResource(Resource):
    def __init__(self,):
        super(ImageResource, self).__init__()
        self.resourceName = 'image'

        self.route('GET', (), self.find)
        self.route('GET', (':id',), self.getImage)
        self.route('GET', (':id', 'thumbnail'), self.thumbnail)
        self.route('GET', (':id', 'download'), self.download)

        self.route('POST', (':id', 'flag'), self.flag)
        self.route('POST', (':id', 'segment'), self.doSegmentation)

        self.route('GET', (':id', 'segmentation', 'latest', 'thumbnail'), self.segmentationThumbnail)


    @describeRoute(
        Description('Return a list of lesion images.')
        .pagingParams(defaultSort='lowerName')
        .param('datasetId', 'The ID of the dataset to use.', required=True)
        .errorResponse()
    )
    @access.public
    def find(self, params):
        self.requireParams('datasetId', params)
        user = self.getCurrentUser()
        limit, offset, sort = self.getPagingParameters(params, 'lowerName')

        dataset = self.model('dataset', 'isic_archive').load(
            id=params['datasetId'], user=user, level=AccessType.READ, exc=True)
        return [
            {
                field: image[field]
                for field in
                self.model('image', 'isic_archive').summaryFields
            }
            for image in
            self.model('dataset', 'isic_archive').childImages(
                dataset, limit=limit, offset=offset, sort=sort)
        ]


    @describeRoute(
        Description('Return an image\'s details.')
        .param('id', 'The ID of the image.', paramType='path')
        .errorResponse('ID was invalid.')
    )
    @access.public
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def getImage(self, image, params):
        return self.model('image', 'isic_archive').filter(
            image, self.getCurrentUser())


    @describeRoute(
        Description('Return an image\'s thumbnail.')
        .param('id', 'The ID of the image.', paramType='path')
        .param('width', 'The desired width for the thumbnail.',
               paramType='query', required=False, default=256)
        .errorResponse('ID was invalid.')
    )
    @access.cookie
    @access.public
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def thumbnail(self, image, params):
        # TODO: fix width to be used
        width = int(params.get('width', 256))

        tileData, tileMime = self.model('image_item', 'large_image').getTile(
            image, 0, 0, 0
        )
        # TODO: crop padding from tileData

        cherrypy.response.headers['Content-Type'] = tileMime
        return lambda: tileData


    @describeRoute(
        Description('Download an image\'s high-quality original binary data.')
        .param('id', 'The ID of the image.', paramType='path')
        .param('contentDisposition', 'Specify the Content-Disposition response '
               'header disposition-type value', required=False,
               enum=['inline', 'attachment'])
        .errorResponse('ID was invalid.')
    )
    @access.cookie
    @access.public
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def download(self, image, params):
        contentDisp = params.get('contentDisposition', None)
        if contentDisp is not None and \
                contentDisp not in {'inline', 'attachment'}:
            raise RestException('Unallowed contentDisposition type "%s".' %
                                contentDisp)

        original_file = self.model('image', 'isic_archive').originalFile(image)
        file_stream = self.model('file').download(original_file, headers=True)

        # TODO: replace this after https://github.com/girder/girder/pull/1123
        if contentDisp == 'inline':
            cherrypy.response.headers['Content-Disposition'] = \
                'inline; filename="%s"' % original_file['name']
        return file_stream


    @describeRoute(
        Description('Flag an image with a problem.')
        .param('id', 'The ID of the image.', paramType='path')
        .errorResponse('ID was invalid.')
    )
    @access.user
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def flag(self, image, params):
        body_json = self.getBodyJson()
        self.requireParams(('reason',), body_json)

        self.model('image', 'isic_archive').flag(
            image, body_json['reason'], self.getCurrentUser())

        return {'status': 'success'}


    @describeRoute(
        Description('Run and return a new semi-automated segmentation.')
        .param('id', 'The ID of the image.', paramType='path')
        .param('seed', 'The X, Y coordinates of a segmentation seed point.',
               paramType='body')
        .param('tolerance',
               'The intensity tolerance value for the segmentation.',
               paramType='body')
        .errorResponse('ID was invalid.')
    )
    @access.user
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def doSegmentation(self, image, params):
        body_json = self.getBodyJson()
        self.requireParams(('seed', 'tolerance'), body_json)

        # validate parameters
        seed_coord = body_json['seed']
        if not (
            isinstance(seed_coord, list) and
            len(seed_coord) == 2 and
            all(isinstance(value, int) for value in seed_coord)
        ):
            raise RestException('Submitted "seed" must be a coordinate pair.')

        tolerance = body_json['tolerance']
        if not isinstance(tolerance, int):
            raise RestException('Submitted "tolerance" must be an integer.')

        try:
            contour_feature = self.model('image', 'isic_archive').doSegmentation(
                image, seed_coord, tolerance)
        except GirderException as e:
            raise RestException(e.message)

        return contour_feature


    @describeRoute(Description(''))
    @access.cookie
    @access.public
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def segmentationThumbnail(self, image, params):
        from bson import ObjectId
        real_image = self.model('image', 'isic_archive').find({
            'baseParentId': ObjectId('55943cff9fc3c13155bcad5e'),
            'name': image['name']}
        )
        assert real_image.count() == 1
        image = real_image.next()
        image_data = self.model('image', 'isic_archive').imageData(image)

        import six
        from PIL import Image as PIL_Image, ImageDraw as PIL_ImageDraw
        from girder.constants import SortDir
        import numpy
        from ..models.segmentation_helpers.scikit import ScikitSegmentationHelper
        from ..models.segmentation_helpers.opencv import OpenCVSegmentationHelper
        segmentation = self.model('segmentation', 'isic_archive').find(
            query={'imageId': image['_id']},
            sort=[('created', SortDir.DESCENDING)]
        ).next()

        segmentation_coords = segmentation['lesionBoundary']['geometry']['coordinates'][0]

        filtered_segmentation_coords = OpenCVSegmentationHelper._maskToContour(
            OpenCVSegmentationHelper._binaryOpening(
                ScikitSegmentationHelper._contourToMask(image_data, segmentation_coords).astype(numpy.uint8),
                element_shape='circle',
                element_radius=5
            )
        )

        width = int(params.get('width', 256))


        pil_image_data = PIL_Image.fromarray(image_data)
        pil_draw = PIL_ImageDraw.Draw(pil_image_data)

        pil_draw.line(
            list(six.moves.map(tuple, segmentation_coords)),
            fill=(255, 0, 0),
            width=5
        )
        pil_draw.line(
            list(six.moves.map(tuple, filtered_segmentation_coords)),
            fill=(0, 255, 0),
            width=5
        )

        output_image_data = six.BytesIO()
        factor = pil_image_data.size[0] / float(width)
        pil_image_data.resize((
            int(pil_image_data.size[0] / factor),
            int(pil_image_data.size[1] / factor)
        )).save(output_image_data, format='jpeg')

        cherrypy.response.headers['Content-Type'] = 'image/jpeg'
        return output_image_data.getvalue
