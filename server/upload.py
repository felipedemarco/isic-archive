#!/usr/bin/env python
# -*- coding: utf-8 -*-

import csv
import mimetypes
import os
import shutil
import sys
import subprocess
import tempfile
import zipfile

from girder.models.model_base import ValidationException
from girder.models.notification import ProgressState
from girder.utility import assetstore_utilities
from girder.utility.model_importer import ModelImporter
from girder.utility.progress import ProgressContext

from .provision_utility import getAdminUser


class TempDir(object):
    def __init__(self):
        pass

    def __enter__(self):
        assetstore = ModelImporter.model('assetstore').getCurrent()
        assetstore_adapter = assetstore_utilities.getAssetstoreAdapter(assetstore)
        try:
            self.temp_dir = tempfile.mkdtemp(dir=assetstore_adapter.tempDir)
        except (AttributeError, OSError):
            self.temp_dir = tempfile.mkdtemp()
        return self.temp_dir

    def __exit__(self, exc_type, exc_val, exc_tb):
        shutil.rmtree(self.temp_dir)


class ZipFileOpener(object):
    def __init__(self, zip_file_path):
        self.zip_file_path = zip_file_path
        # TODO: check for "7z" command

    def __enter__(self):
        try:
            return self._defaultUnzip()
        except zipfile.BadZipfile:
            return self._fallbackUnzip()

    def __exit__(self, exc_type, exc_val, exc_tb):
        pass

    def _defaultUnzip(self):
        zip_file = zipfile.ZipFile(self.zip_file_path)

        # filter out directories and count real files
        file_list = list()
        for original_file in zip_file.infolist():
            original_file_relpath = original_file.filename
            original_file_relpath.replace('\\', '/')
            original_file_name = os.path.basename(original_file_relpath)
            if not original_file_name or not original_file.file_size:
                # file is probably a directory, skip
                continue
            file_list.append((original_file, original_file_relpath))
        return self._defaultUnzipIter(zip_file, file_list), len(file_list)


    def _defaultUnzipIter(self, zip_file, file_list):
        with TempDir() as temp_dir:
            for original_file, original_file_relpath in file_list:
                original_file_name = os.path.basename(original_file_relpath)
                temp_file_path = os.path.join(temp_dir, original_file_name)
                with open(temp_file_path, 'wb') as temp_file_obj:
                    shutil.copyfileobj(
                        zip_file.open(original_file),
                        temp_file_obj
                    )
                yield temp_file_path, original_file_relpath
                os.remove(temp_file_path)
            zip_file.close()


    def _fallbackUnzip(self):
        with TempDir() as temp_dir:
            unzip_command = (
                '7z',
                'x',
                '-y',
                '-o%s' % temp_dir,
                self.zip_file_path
            )
            try:
                with open(os.devnull, 'rb') as null_in,\
                        open(os.devnull, 'wb') as null_out:
                    subprocess.check_call(unzip_command,
                        stdin=null_in, stdout=null_out, stderr=subprocess.STDOUT)
            except subprocess.CalledProcessError:
                self.__exit__(*sys.exc_info())
                raise

            file_list = list()
            for temp_dir_path, _, temp_file_names in os.walk(temp_dir):
                for temp_file_name in temp_file_names:
                    temp_file_path = os.path.join(temp_dir_path, temp_file_name)
                    original_file_relpath = os.path.relpath(temp_file_path, temp_dir)
                    file_list.append((temp_file_path, original_file_relpath))
            return iter(file_list), len(file_list)


def handleZip(images_folder, user, zip_file):
    Image = ModelImporter.model('image', 'isic_archive')

    # Get full path of zip file in assetstore
    assetstore = ModelImporter.model('assetstore').getCurrent()
    assetstore_adapter = assetstore_utilities.getAssetstoreAdapter(assetstore)
    full_path = assetstore_adapter.fullPath(zip_file)

    with ZipFileOpener(full_path) as (file_list, file_count):
        with ProgressContext(
                on=True,
                user=user,
                title='Processing "%s"' % zip_file['name'],
                total=file_count,
                state=ProgressState.ACTIVE,
                current=0) as progress:

            for original_file_path, original_file_relpath in file_list:
                original_file_name = os.path.basename(original_file_relpath)

                progress.update(
                    increment=1,
                    message='Extracting "%s"' % original_file_name)

                image_item = Image.createImage(
                    creator=user,
                    parentFolder=images_folder
                )
                Image.setMetadata(image_item, {
                    'originalFilename': os.path.splitext(original_file_name)[0]
                })

                # upload original image
                image_mimetype = mimetypes.guess_type(original_file_name)[0]
                with open(original_file_path, 'rb') as original_file_obj:
                    ModelImporter.model('upload').uploadFromFile(
                        obj=original_file_obj,
                        size=os.path.getsize(original_file_path),
                        name='%s%s' % (
                            image_item['name'],
                            os.path.splitext(original_file_name)[1].lower()
                        ),
                        parentType='item',
                        parent=image_item,
                        user=user,
                        mimeType=image_mimetype,
                    )
                # reload image_item, since its 'size' has changed in the database
                image_item = Image.load(image_item['_id'], force=True)

                image_data = Image.imageData(image_item)
                image_item['meta']['acquisition']['pixelsY'] = image_data.shape[0]
                image_item['meta']['acquisition']['pixelsX'] = image_data.shape[1]
                Image.save(image_item)


def handleImage(image_item, user, license):
    Image = ModelImporter.model('image', 'isic_archive')
    Item = ModelImporter.model('item')

    assetstore = ModelImporter.model('assetstore').getCurrent()
    assetstore_adapter = assetstore_utilities.getAssetstoreAdapter(assetstore)

    with TempDir() as temp_dir:
        converted_file_name = '%s.tif' % image_item['name']
        converted_file_path = os.path.join(temp_dir, converted_file_name)

        image_item_files = Item.childFiles(image_item)
        full_path = assetstore_adapter.fullPath(image_item_files[0])

        convert_command = (
            '/usr/local/bin/vips',
            'tiffsave',
            '\'%s\'' % full_path,
            '\'%s\'' % converted_file_path,
            '--compression', 'jpeg',
            '--Q', '90',
            '--tile',
            '--tile-width', '256',
            '--tile-height', '256',
            '--pyramid',
            '--bigtiff',
        )
        # TODO: subprocess.check_call is causing the whole application to crash
        os.popen(' '.join(convert_command))
        # try:
        #     with open(os.devnull, 'rb') as null_in,\
        #             open(os.devnull, 'wb') as null_out:
        #         subprocess.check_call(convert_command,
        #             stdin=null_in, stdout=null_out, stderr=subprocess.STDOUT)
        # except subprocess.CalledProcessError:
        #     try:
        #         os.remove(converted_file_path)
        #     except OSError:
        #         pass
        #     continue

        # upload converted image
        with open(converted_file_path, 'rb') as converted_file_obj:
            converted_file = ModelImporter.model('upload').uploadFromFile(
                obj=converted_file_obj,
                size=os.path.getsize(converted_file_path),
                name=converted_file_name,
                parentType='item',
                parent=image_item,
                user=user,
                mimeType='image/tiff',
            )
        os.remove(converted_file_path)
        # reload image_item, since its 'size' has changed in the database
        image_item = Image.load(image_item['_id'], force=True)

        image_item['largeImage'] = {
            'fileId': converted_file['_id'],
            'sourceName': 'tiff'
        }
        if license:
            image_item['license'] = license

        Image.save(image_item)


def handleCsv(dataset_folder, user, csv_file):
    dataset_folder_ids = [
        folder['_id']
        for folder in ModelImporter.model('folder').find({
            'name': dataset_folder['name']
        })
    ]
    # TODO: ensure that dataset_folder_ids are UDA folders / in UDA phases

    assetstore = ModelImporter.model('assetstore').getCurrent()
    assetstore_adapter = assetstore_utilities.getAssetstoreAdapter(assetstore)
    full_path = assetstore_adapter.fullPath(csv_file)

    parse_errors = list()
    with open(full_path, 'rUb') as upload_file_obj,\
        ProgressContext(
            on=True,
            user=user,
            title='Processing "%s"' % csv_file['name'],
            state=ProgressState.ACTIVE,
            message='Parsing CSV') as progress:

        # csv.reader(csvfile, delimiter=',', quotechar='"')
        csv_reader = csv.DictReader(upload_file_obj)

        for filenameField in csv_reader.fieldnames:
            if filenameField.lower() == 'filename':
                break
        else:
            raise ValidationException('No "filename" field found in CSV.')

        for csv_row in csv_reader:
            filename = csv_row.pop(filenameField, None)
            if not filename:
                parse_errors.append('No "filename" field in row %d' % csv_reader.line_num)
                continue

            # TODO: require 'user' to match image creator?
            # TODO: index on meta.originalFilename?
            image_items = ModelImporter.model('image', 'isic_archive').find({
                'meta.originalFilename': filename,
                'folderId': {'$in': dataset_folder_ids}
            })
            if not image_items.count():
                parse_errors.append(
                    'No image found with original filename "%s"' % filename)
                continue
            elif image_items.count() > 1:
                parse_errors.append(
                    'Multiple images found with original filename "%s"' % filename)
                continue
            else:
                image_item = image_items.next()

            clinical_metadata = image_item['meta']['clinical']
            clinical_metadata.update(csv_row)
            ModelImporter.model('image', 'isic_archive').setMetadata(image_item, {
                'clinical': clinical_metadata
            })

    if parse_errors:
        # TODO: eventually don't store whole string in memory
        parse_errors_str = '\n'.join(parse_errors)

        parent_item = ModelImporter.model('item').load(csv_file['itemId'], force=True)

        upload = ModelImporter.model('upload').createUpload(
            user=getAdminUser(),
            name='parse_errors.txt',
            parentType='item',
            parent=parent_item,
            size=len(parse_errors_str),
            mimeType='text/plain',
        )
        ModelImporter.model('upload').handleChunk(
            upload, parse_errors_str)
