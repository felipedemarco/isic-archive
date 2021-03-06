/**
 * Copyright 2016 Kitware Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

module.exports = function (grunt) {
    var path = require('path');

    // This gruntfile is only designed to be used with girder's build system.
    // Fail if grunt is executed here.
    if (path.resolve(__dirname) === path.resolve(process.cwd())) {
        grunt.fail.fatal('To build isic_archive, run grunt from Girder\'s root directory');
    }

    grunt.config.merge({
        // This step will fail if '<%= pluginDir %>/isic_archive' is a symlink
        /*
        symlink: {
            'plugin-isic_archive-bower': {
                files: [{
                    src: ['<%= pluginDir %>/isic_archive/bower_components'],
                    dest: '<%= pluginDir %>/isic_archive/web_client/extra/bower_components'
                }]
            }
        },

        init: {
            'symlink:plugin-isic_archive-bower': {
                dependencies: ['shell:plugin-isic_archive']
            }
        }
        */
    });

    var fs = require('fs');
    var defaultTasks = [];

    // Since this is an external web app in a plugin,
    // it handles building itself
    //
    // It is not included in the plugins being built by virtue of
    // the web client not living in web_client, but rather web_external
    var configureIsicArchive = function () {
        var pluginName = "isic_archive";
        var pluginDir = "plugins/isic_archive";
        var staticDir = 'clients/web/static/built/plugins/' + pluginName;
        var sourceDir = "web_external";

        if (!fs.existsSync(staticDir)) {
            fs.mkdirSync(staticDir);
        }

        var jadeDir = pluginDir + '/' + sourceDir + '/templates';
        if (fs.existsSync(jadeDir)) {

            var files = {};
            files[staticDir + '/isic_archive_templates.js'] = [jadeDir + '/**/*.jade'];
            grunt.config.set('jade.' + pluginName, {
                files: files
            });
            grunt.config.set('jade.' + pluginName + '.options', {
                namespace: 'isic.templates'
            });
            grunt.config.set('watch.jade_' + pluginName + '_app', {
                files: [jadeDir + '/**/*.jade'],
                tasks: ['jade:' + pluginName, 'uglify:' + pluginName]
            });
            defaultTasks.push('jade:' + pluginName);
        }

        var cssDir = pluginDir + '/' + sourceDir + '/stylesheets';
        if (fs.existsSync(cssDir)) {
            var files = {};
            files[staticDir + '/isic_archive.min.css'] = [cssDir + '/**/*.styl'];
            grunt.config.set('stylus.' + pluginName, {
                files: files
            });
            grunt.config.set('watch.stylus_' + pluginName + '_app', {
                files: [cssDir + '/**/*.styl'],
                tasks: ['stylus:' + pluginName]
            });
            defaultTasks.push('stylus:' + pluginName);
        }

        var jsDir = pluginDir + '/' + sourceDir + '/js';
        if (fs.existsSync(jsDir)) {
            var files = {};
            // name this isic_archive.min.js instead of plugin.min.js
            // so that girder app won't load isic_archive, which
            // should only be loaded as a separate web app running as isic_archive
            files[staticDir + '/isic_archive.min.js'] = [
                jsDir + '/init.js',
                staticDir + '/isic_archive_templates.js',
                jsDir + '/isic_archive-version.js',
                jsDir + '/view.js',
                jsDir + '/app.js',
                jsDir + '/models/**/*.js',
                jsDir + '/collections/**/*.js',
                jsDir + '/views/**/*.js'
            ];
            files[staticDir + '/main.min.js'] = [
                jsDir + '/main.js'
            ];
            grunt.config.set('uglify.' + pluginName, {
                files: files
            });
            grunt.config.set('watch.js_' + pluginName + '_app', {
                files: [jsDir + '/**/*.js'],
                tasks: ['uglify:' + pluginName]
            });
            defaultTasks.push('uglify:' + pluginName);
        }

        var extraDir = pluginDir + '/' + sourceDir + '/extra';
        if (fs.existsSync(extraDir)) {
            grunt.config.set('copy.' + pluginName, {
                expand: true,
                cwd: pluginDir + '/' + sourceDir,
                src: ['extra/**'],
                dest: staticDir
            });
            grunt.config.set('watch.copy_' + pluginName, {
                files: [extraDir + '/**/*'],
                tasks: ['copy:' + pluginName]
            });
            defaultTasks.push('copy:' + pluginName);
        }
    };

    configureIsicArchive();
    grunt.registerTask('isic_archive-web', defaultTasks);

};
