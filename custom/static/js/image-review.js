/**
 * Created by stonerri on 3/13/14.
 */

'use strict';
/* global console */

// Initialization of angular root application
var review_app = angular.module('DermApp', ['ngSanitize', 'mousetrap']);

review_app.config(function($httpProvider) {
  $httpProvider.defaults.xsrfCookieName = 'girderToken';
  $httpProvider.defaults.xsrfHeaderName = 'Girder-Token';
});

var appController = review_app.controller('ApplicationController', ['$scope', '$rootScope', '$timeout', '$http',
    function ($scope, $rootScope, $timeout, $http) {

        $("#angular_id").height(window.innerHeight - 80 - 100);
        $("#gridcontainer").height(window.innerHeight - 100 - 100);

        var urlvals = window.location.pathname.split('/');
        var folder_id = urlvals[urlvals.length - 1];

        $scope.images_url = '/api/v1/item?folderId=' + folder_id;
        $scope.folder_url = '/api/v1/folder/' + folder_id;
        $scope.taskcomplete_url = '/api/v1/uda/task/qc/' + folder_id + '/complete';

        $scope.getFolderInfo = function () {
            $http.get($scope.folder_url).then(function (response) {
                $scope.folder_details = response.data;
            });
        };

        $scope.hover_image = undefined;
        $scope.flagged_list = {};
        $scope.image_list = [];
        $scope.show_flags = false;

//        $scope.download = function(){
//
//            var flags = $scope.flagged_list;
//            flags['user'] = $scope.user_email;
//
//            var json = JSON.stringify(flags);
//            var blob = new Blob([json], {type: "application/json"});
//
//            // save as json to local computer
//            saveAs(blob, "flagged_images.json");
//        };


        $scope.getImages = function () {

            $http.get($scope.images_url).then(function (response) {

                console.log(response);
                $scope.flagged_list = {};

                var temporary_array = [];

                response.data.forEach(function(image){

                    var simple_rep = image;
                    simple_rep['thumbnail'] = '/api/v1/item/' + image['_id'] + '/thumbnail';
                    simple_rep['title'] = image['name'];

                    temporary_array.push(simple_rep);

                });

                $scope.image_list = temporary_array;
                $scope.show_flags = true;

            });
        };


        $scope.clearFlagged = function () {
            var flagged_images = [];
            $scope.flagged_list.forEach(function (flagged_image) {
                flagged_images.push(flagged_image._id);
            });

            var payload = {
                flagged: flagged_images,
                good : []
            };

            $http.post($scope.taskcomplete_url, angular.toJson(payload)).then(function(response){
                if (response.status === 200) {
                    $scope.getImages();
                    $scope.getFolderInfo();
                }
            });
        };


        $scope.submitAll = function () {
            var flagged_images = [];
            $scope.flagged_list.forEach(function (flagged_image) {
                flagged_images.push(flagged_image._id);
            });

            var images_to_accept = [];
            for (var image_index in $scope.image_list){
                if( !(image_index in $scope.flagged_list) ) {
                    images_to_accept.push($scope.image_list[image_index]._id);
                }
            }

            var payload = {
                flagged: flagged_images,
                good : images_to_accept,
            };

            $http.post($scope.taskcomplete_url, angular.toJson(payload)).then(function(response){
                if (response.status === 200){
                    $scope.getImages();
                }
            });
        };



        $scope.mouse = {
            '.' : $scope.nextSet,
            ',' : $scope.previousSet
        };


        $scope.flagged = function(index){
            var t = $scope.flagged_list[index];
            return (t !== undefined);
        };

        $scope.getOffset = function(index){
            return $scope.start + index;
        };


        $scope.toggleFlag = function(index){

            var t = $scope.flagged_list[index];

            if (t !== undefined) {

                delete $scope.flagged_list[index];
            }
            else {
                $scope.flagged_list[index] = $scope.image_list[index];
            }

            console.log($scope.flagged_list);
        };



        $scope.imageHasAnnotations = function(index){
            return $scope.image_list[index].annotation > 0;
        };

        $scope.safeApply = function( fn ) {
            var phase = this.$root.$$phase;
            if(phase === '$apply' || phase === '$digest') {
                if(fn) { fn(); }
            } else {
                this.$apply(fn);
            }
        };

        $scope.getFolderInfo();
        $scope.getImages();
    }
]);



