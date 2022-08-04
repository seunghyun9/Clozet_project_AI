
import datetime
from fileinput import filename
from fastapi import FastAPI, HTTPException, File, UploadFile
from typing import Optional
from fastapi.middleware.cors import CORSMiddleware


from typing import List
import os
from spec import Spec

app = FastAPI()
origins = [ "http://localhost:8000",]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

    

@app.get("/")
def now():
    return {"Now":datetime.datetime.now().strftime('%Y-%m-%d')}


@app.post("/files/")
async def create_files(files: List[bytes] = File(...)):
    return {"file_sizes": [len(file) for file in files]}


@app.post("/uploadfiles")
async def create_upload_files(files: List[UploadFile] = File(...)):
    print('들어옴1')
    UPLOAD_DIRECTORY = "./"
    for file in files:
        print('들어옴2')
        contents = await file.read()
        with open(os.path.join(UPLOAD_DIRECTORY, file.filename), "wb") as fp:
            fp.write(contents)
        print(file.filename)
    #a= {"filenames": [file.filename for file in files]}
    #print(a)
    #self.spec(a)
    a = {"filenames": [file.filename for file in files]}
    b= [file.filename for file in files][0]
    print(b)
    print('들어옴3')
    return Spec.service(filename=b)




@app.get("/item/")
async def root():
    return {"message":"Hello World"}




@app.get("/item/")
async def root():
    return {"message":"Hello World"}


