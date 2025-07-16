from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive


gauth = GoogleAuth()
#creates local webserver and auto handles authentication
gauth.LocalWebserverAuth()
drive = GoogleDrive(gauth)

def updateFile() -> None:
    fileList = drive.ListFile({'q': "'root' in parents and trashed=false"}).GetList()
    for file1 in fileList:
        if file1['id'] == "12ChYdknEK6Mfw9gfK2IYG4RxONdf35BpIHBunOv0Too":
            file2 = file1

    filetoCSV = drive.CreateFile({'id': file2['id']})
    filetoCSV.GetContentFile('file.csv', mimetype='text/csv')