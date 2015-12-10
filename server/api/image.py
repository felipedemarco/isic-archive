#!/usr/bin/env python
# -*- coding: utf-8 -*-

import datetime

import cherrypy

from girder.api import access
from girder.api.rest import Resource, RestException, loadmodel
from girder.api.describe import Description, describeRoute
from girder.constants import AccessType
from girder.models.model_base import AccessException

from ..image_processing import fillImageGeoJSON
from ..provision_utility import _ISICCollection


class ImageResource(Resource):
    def __init__(self,):
        super(ImageResource, self).__init__()
        self.resourceName = 'image'

        self.route('GET', (), self.find)
        self.route('GET', (':id',), self.getImage)
        self.route('GET', (':id', 'thumbnail'), self.thumbnail)
        self.route('GET', (':id', 'download'), self.download)

        self.route('POST', (':id', 'flag'), self.flag)

        # TODO: change to GET
        self.route('POST', (':id', 'segment-boundary'), self.segmentBoundary)


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
        .errorResponse('ID was invalid.')
    )
    @access.cookie
    @access.public
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def thumbnail(self, image, params):
        width = int(params.get('width', 256))
        thumbnail_url = self.model('image', 'isic_archive').tileServerURL(
            image, width=width)
        raise cherrypy.HTTPRedirect(thumbnail_url, status=307)


    @describeRoute(
        Description('Download an image\'s high-quality original binary data.')
        .param('id', 'The ID of the image.', paramType='path')
        .errorResponse('ID was invalid.')
    )
    @access.cookie
    @access.public
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def download(self, image, params):
        original_file = self.model('image', 'isic_archive').originalFile(image)
        return self.model('file').download(original_file, headers=True)


    @describeRoute(
        Description('Flag an image with a problem.')
        .param('id', 'The ID of the image.', paramType='path')
        .errorResponse('ID was invalid.')
    )
    @access.cookie
    @access.user
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def flag(self, image, params):
        body_json = self.getBodyJson()
        self.requireParams(('reason',), body_json)

        # TODO: change to use direct permissions on the image
        if not any(
            self.model('group').findOne(
                {'name': groupName}
            )['_id'] in self.getCurrentUser()['groups']
            for groupName in
            ['Phase 0', 'Phase 1a', 'Phase 1b']
        ):
            raise AccessException('User does not have permission to flag this image.')

        image_dataset= self.model('dataset', 'isic_archive').load(
            image['folderId'], force=True)

        flagged_folder = self.model('folder').findOne({
            'parentId': self.model('collection').findOne({'name': 'Phase 0'})['_id'],
            'name': 'flagged'
        })
        phase0_flagged_images = _ISICCollection.createFolder(
            name=image_dataset['name'],
            description='',
            parent=flagged_folder,
            parent_type='folder'
        )

        flag_metadata = {
            'flaggedUserId': self.getCurrentUser()['_id'],
            'flaggedTime': datetime.datetime.utcnow(),
            'flaggedReason': body_json['reason'],
        }
        self.model('item').setMetadata(image, flag_metadata)
        # TODO: deal with any existing studies with this image
        self.model('item').move(image, phase0_flagged_images)

        return {'status': 'success'}


    @describeRoute(
        Description('Return an image\'s boundary segmentation.')
        # .responseClass('Image')
        .param('id', 'The ID of the image.', paramType='path')
        .errorResponse('ID was invalid.')
    )
    @access.user
    @loadmodel(model='image', plugin='isic_archive', level=AccessType.READ)
    def segmentBoundary(self, image, params):
        body_json = self.getBodyJson()
        self.requireParams(('seed', 'tolerance'), body_json)

        # validate parameters
        seed_point = body_json['seed']
        if not (
            isinstance(seed_point, list) and
            len(seed_point) == 2 and
            all(isinstance(value, int) for value in seed_point)
        ):
            raise RestException('Submitted "seed" must be a coordinate pair.')

        tolerance = body_json['tolerance']
        if not isinstance(tolerance, int):
            raise RestException('Submitted "tolerance" must be an integer.')

        image_data = self.model('image', 'isic_archive').binaryImageRaw(image)

        results = fillImageGeoJSON(
            image_data=image_data,
            seed_point=seed_point,
            tolerance=tolerance
        )

        return results
        # return json.dumps(results)
