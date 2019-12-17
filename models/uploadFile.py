
UPLOAD_FOLDER = '/Users/nawafalsuwailem/PycharmProjects/mysite/UPLOAD_FOLDER'
ALLOWED_EXTENSIONS = {'csv'}

def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS



