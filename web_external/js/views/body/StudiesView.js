isic.views.StudiesView = isic.View.extend({
    // TODO refactor
    events: {
        'show.bs.collapse .isic-listing-panel-collapse': function (event) {
            var target = $(event.target);
            target.parent().find('.icon-right-open').removeClass('icon-right-open').addClass('icon-down-open');

            var viewIndex = window.parseInt(target.attr('data-model-index'), 10);
            var viewContainer = target.find('.isic-listing-panel-body');
            this.renderStudy(viewIndex, viewContainer);
        },
        'hide.bs.collapse .isic-listing-panel-collapse': function (event) {
            $(event.target).parent().find('.icon-down-open').removeClass('icon-down-open').addClass('icon-right-open');
        },
        'click .isic-study-add-button': function () {
            isic.router.navigate('createStudy', {trigger: true});
        }
    },

    initialize: function (settings) {
        this.loaded = false;
        this.studyAdmin = false;
        this.studies = new isic.collections.StudyCollection();

        // Check whether user has permission to create studies
        // TODO re-render open study listings if this changes
        var studyModel = new isic.models.StudyModel();
        studyModel.isAdministrator(girder.currentUser).then(_.bind(function (studyAdmin) {
            if (this.studyAdmin !== studyAdmin) {
                this.studyAdmin = studyAdmin;
                this.render();
            }
        }, this));

        this.studies.once('g:changed', function () {
            this.loaded = true;
            this.render();
        }, this).fetch();

        this.render();
    },

    render: function () {
        this.$el.html(isic.templates.studiesPage({
            title: 'Annotation Studies',
            models: this.studies.models,
            loaded: this.loaded,
            studyAdmin: this.studyAdmin
        }));

        // Display loading indicator
        if (!this.loaded) {
            new girder.views.LoadingAnimation({
                el: this.$('.isic-listing-loading-animation-container'),
                parentView: this
            }).render();
        }

        this.$('.isic-tooltip').tooltip({
            delay: 100
        });

        return this;
    },

    renderStudy: function (index, container) {
        if (container.children().length === 0) {
            var studyId = this.studies.at(index).id;

            // Display loading indicator
            new girder.views.LoadingAnimation({
                el: container,
                parentView: this
            }).render();

            new isic.views.StudyView({ // eslint-disable-line no-new
                el: container,
                id: studyId,
                studyAdmin: this.studyAdmin,
                parentView: this
            });
        }
    }
});

isic.router.route('studies', 'studies', function (id) {
    girder.events.trigger('g:navigateTo', isic.views.StudiesView);
});
