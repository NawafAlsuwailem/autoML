from flask import Flask,render_template, flash, request, redirect, url_for, send_from_directory
import os
from werkzeug.utils import secure_filename
from werkzeug.middleware.shared_data import SharedDataMiddleware
from models.data_understanding import *
from models.uploadFile import *
from models.eda import *
from models.preprocessing import *
from flask import Flask, session
from models.modelling import *
from sklearn.externals import joblib

""" 
    Upload file code 
"""

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.secret_key = os.urandom(2000)
app.add_url_rule('/uploads/<filename>', 'uploaded_file',
                 build_only=True)
app.wsgi_app = SharedDataMiddleware(app.wsgi_app, {
    '/uploads':  app.config['UPLOAD_FOLDER']
})


@app.route('/', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        if 'file' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return render_template('upload.html', message="no file selected")
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            return redirect(url_for('data_understanding',
                                    filename=filename))
    return render_template('upload.html')


""" 
    Data => neutral 
    Display file code / data understanding
"""


@app.route("/data_understanding/<filename>", methods=['GET', 'POST'])
def data_understanding(filename):
    if request.method == "GET":
        dataframe = pd.read_csv(UPLOAD_FOLDER + "/" + filename, sep=";|,", index_col=False)
        dataframe.columns = [i.lower() for i in dataframe.columns]
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == dataframe.shape[0]:
                del dataframe[column]
        columns = dataframe.columns
        shape = dataframe.shape
        data_information = process_content_info(dataframe)
        data_describe = pd.DataFrame(dataframe.describe())
        bars = get_feature_stats(dataframe).values
        return render_template("data_understanding.html",
                               name=filename,
                               data=dataframe.head(10).to_html(),
                               columns=columns,
                               shape=shape,
                               data_information=data_information.to_html(),
                               data_describe=data_describe.to_html(),
                               plots=bars,
                               )
    else:
        target_feature = request.form.get('target_feature')
        outlier = request.form.get('outlier')
        null = request.form.get('null')
        return redirect(url_for('eda',
                                filename=filename,
                                target_feature=target_feature,
                                outlier=outlier,
                                null=null,
                                ))


""" 
    Data => preporcessed
    Display EDA 
"""


@app.route("/eda/<filename>, <target_feature>, <outlier>,<null>", methods=['GET', 'POST'])
def eda(filename, target_feature, outlier, null):
    if request.method == "GET":
        dataframe = pd.read_csv(UPLOAD_FOLDER + "/" + filename, sep=";|,", index_col=False)
        dataframe.columns = [i.lower() for i in dataframe.columns]
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == dataframe.shape[0]:
                del dataframe[column]
        columns = dataframe.columns
        dataframe = deal_null(dataframe, null)
        dataframe = apply_label_encoder(dataframe)
        dataframe = deal_outlier(dataframe, outlier)
        feature_impor_details, feature_by_importance, feature_import_chart = get_feature_importance(dataframe, target_feature)
        heatmap = get_heatmap(dataframe)
        scatter_matrix = get_scatter_matrix(dataframe, target_feature)
        data_dists = get_data_dist(dataframe,  target_feature).values
        return render_template("eda.html",
                               dataframe=dataframe.sample(10).to_html(),
                               columns=columns.drop(target_feature),
                               outlier=outlier,
                               feature_impor_details=feature_impor_details.to_html(),
                               feature_by_importance=feature_by_importance,
                               feature_import_chart=feature_import_chart,
                               heatmap=heatmap,
                               scatter_matrix=scatter_matrix,
                               data_dists=data_dists,
        )
    else:
        test_size = request.form.get('test_size')
        selected_features = request.form.getlist('check_list')
        session["selected_features"] = selected_features
        session["target_feature"] = target_feature
        session["outlier"] = outlier
        session["null"] = null
        session["test_size"] = test_size
        return redirect(url_for('modelling',
                                filename=filename,
                                ))


@app.route("/modelling/<filename>", methods=['GET'])
def modelling(filename):
    if request.method == "GET":
        selected_features = session.get('selected_features')
        target_feature = session.get('target_feature')
        outlier = session.get('outlier')
        null = session.get('null')
        test_size = session.get('test_size')
        dataframe = pd.read_csv(UPLOAD_FOLDER + "/" + filename, sep=";|,", index_col=False)
        dataframe.columns = [i.lower() for i in dataframe.columns]
        for column in dataframe.columns:
            if len(dataframe[column].unique()) == dataframe.shape[0]:
                del dataframe[column]
        columns = dataframe.columns
        baseline = get_baseline(dataframe, target_feature)

        categorical_columns = get_categorical_data(dataframe).columns
        dataframe = deal_null(dataframe, null)
        dataframe = apply_label_encoder(dataframe)
        dataframe = deal_outlier(dataframe, outlier)
        X = define_independent_variables(dataframe, target_feature)
        y = define_dependant_variable(dataframe, target_feature)
        X = get_selected_features(X, selected_features)
        X = apply_one_hot_encoder(X, categorical_columns)
        test_size = float(test_size)/10
        X_train, X_test, y_train, y_test = split_data_set(X, y, test_size)
        X_train, X_test = apply_standard_scaling(X_train, X_test)

        # classifier
        ann = get_ann(X_train, y_train)
        knn = get_knn(X_train, y_train)

        # classifier info
        ann_ypredict = get_ypredict(ann, X_test)
        knn_ypredict = get_ypredict(knn, X_test)

        ann_cm = get_confusion_matrix(y_test, ann_ypredict)
        knn_cm = get_confusion_matrix(y_test, knn_ypredict)

        ann_acc = get_acc_score(ann_ypredict, y_test)
        knn_acc = get_acc_score(knn_ypredict, y_test)

        recommended_model = ""
        if ann_acc > knn_acc:
            recommended_model = "ann"
        elif ann_acc < knn_acc:
            recommended_model = "knn"
        else:
            recommended_model = ""

        ann_report = get_classification_report(y_test, ann_ypredict)
        knn_report = get_classification_report(y_test, knn_ypredict)

        ann_filename = 'ann.sav'
        joblib.dump(ann, "ml_models/"+ann_filename)

        return render_template("modelling.html",
                               baseline=baseline.to_html(),
                               ann_report=ann_report.to_html(),
                               knn_report=knn_report.to_html(),
                               recommended_model=recommended_model,
                               )



"""
    Run app
"""
if __name__ == '__main__':
    SESSION_TYPE = "filesystem"
    HOST = os.environ.get('SERVER_HOST', 'localhost')
    try:
        PORT = int(os.environ.get("SERVER_PORT", '5555'))
    except ValueError:
        PORT = 5555
    app.run()
