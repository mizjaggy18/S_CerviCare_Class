# -*- coding: utf-8 -*-

# * Copyright (c) 2009-2018. Authors: see NOTICE file.
# *
# * Licensed under the Apache License, Version 2.0 (the "License");
# * you may not use this file except in compliance with the License.
# * You may obtain a copy of the License at
# *
# *      http://www.apache.org/licenses/LICENSE-2.0
# *
# * Unless required by applicable law or agreed to in writing, software
# * distributed under the License is distributed on an "AS IS" BASIS,
# * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# * See the License for the specific language governing permissions and
# * limitations under the License.

from __future__ import print_function, unicode_literals, absolute_import, division


import sys
import numpy as np
import os
import cytomine
from shapely.geometry import shape, box, Polygon,Point
from shapely import wkt
from glob import glob

from cytomine import Cytomine, models, CytomineJob
from cytomine.models import Annotation, AnnotationTerm, AnnotationCollection, ImageInstanceCollection, Job, JobData, Project, ImageInstance, Property
from cytomine.models.ontology import Ontology, OntologyCollection, Term, RelationTerm, TermCollection

import torch
from torchvision.models import DenseNet

import time
import cv2
import math


from argparse import ArgumentParser
import json
import logging
import logging.handlers
import shutil

__author__ = "WSH Munirah W Ahmad <wshmunirah@gmail.com>"
__version__ = "1.0.1"

def run(cyto_job, parameters):
    logging.info("----- CerviCare-class-Pytorch v%s -----", __version__)
    logging.info("Entering run(cyto_job=%s, parameters=%s)", cyto_job, parameters)

    job = cyto_job.job
    user = job.userJob
    project = cyto_job.project

    terms = TermCollection().fetch_with_filter("project", parameters.cytomine_id_project)
    job.update(status=Job.RUNNING, progress=1, statusComment="Terms collected...")
    print(terms)

    start_time=time.time()

    # ----- load network ----
    modelname = "/models/cervicare-2class_dn21adam_best_model_100ep.pth"
    gpuid = 0

    device = torch.device(gpuid if gpuid!=-2 and torch.cuda.is_available() else 'cpu')
    print("Device: ", device)

    checkpoint = torch.load(modelname, map_location=lambda storage, loc: storage) #load checkpoint to CPU and then put to device https://discuss.pytorch.org/t/saving-and-loading-torch-models-on-2-machines-with-different-number-of-gpu-devices/6666

    model = DenseNet(growth_rate=checkpoint["growth_rate"], block_config=checkpoint["block_config"],
                    num_init_features=checkpoint["num_init_features"], bn_size=checkpoint["bn_size"],
                    drop_rate=checkpoint["drop_rate"], num_classes=checkpoint["num_classes"]).to(device)

    model.load_state_dict(checkpoint["model_dict"])
    model.eval()

    print("Model name: ",modelname)
    # ------------------------

    print(f"Model successfully loaded! Total params: \t{sum([np.prod(p.size()) for p in model.parameters()])}")
    job.update(status=Job.RUNNING, progress=20, statusComment=f"Model successfully loaded!")

    #Select images to process
    images = ImageInstanceCollection().fetch_with_filter("project", project.id)       
    list_imgs = []
    if parameters.cytomine_id_images == 'all':
        for image in images:
            list_imgs.append(int(image.id))
    else:
        list_imgs = parameters.cytomine_id_images
        list_imgs2 = list_imgs.split(',')
        
    print('Print list images:', list_imgs2)
    job.update(status=Job.RUNNING, progress=30, statusComment="Images gathered...")

    #Set working path
    working_path = os.path.join("tmp", str(job.id))
   
    if not os.path.exists(working_path):
        logging.info("Creating working directory: %s", working_path)
        os.makedirs(working_path)
    try:

        id_project=project.id   
        output_path = os.path.join(working_path, "classification_results.csv")
        f= open(output_path,"w+")
        
        #Go over images
        for id_image in list_imgs2:

            print('Current image:', id_image)
            imageinfo=ImageInstance(id=id_image,project=parameters.cytomine_id_project)
            imageinfo.fetch()
            calibration_factor=imageinfo.resolution
            
            roi_annotations = AnnotationCollection(
                terms=[parameters.cytomine_id_cell_term],
                project=parameters.cytomine_id_project,
                image=id_image, #conn.parameters.cytomine_id_image
                showWKT = True,
                includeAlgo=True, 
            )
            roi_annotations.fetch()
            print(roi_annotations)
            #for roi in conn.monitor(roi_annotations, prefix="Running detection on ROI", period=0.1):
            job.update(status=Job.RUNNING, progress=60, statusComment="Running classification on image...")


            roi_numel=len(roi_annotations)
            x=range(1,roi_numel)
            increment=np.multiply(10000,x)

            start_prediction_time=time.time()
            predictions = []
            img_all = []
            pred_all = []
            pred_c0 = 0
            pred_c1 = 0    

            #Go over ROI in this image
            for i, roi in enumerate(roi_annotations):
                
                for inc in increment:
                    if i==inc:
                        shutil.rmtree(roi_path, ignore_errors=True)
                        import gc
                        gc.collect()
                        print("i==", inc)

                # try:

                roi_geometry = wkt.loads(roi.location)

                #Dump ROI image into local PNG file
                roi_path=os.path.join(working_path,str(roi_annotations.project)+'/'+str(roi_annotations.image)+'/'+str(roi.id))
                roi_png_filename=os.path.join(roi_path+'/'+str(roi.id)+'.png')
                print("roi_png_filename: %s" %roi_png_filename)
                is_algo = User().fetch(roi.user).algo
                roi.dump(dest_pattern=roi_png_filename,mask=True,alpha=not is_algo)

                im = cv2.cvtColor(cv2.imread(roi_png_filename),cv2.COLOR_BGR2RGB)
                im = cv2.resize(im,(224,224))
                im = im.reshape(-1,224,224,3)
                output = np.zeros((0,checkpoint["num_classes"]))
                arr_out_gpu = torch.from_numpy(im.transpose(0, 3, 1, 2)).type('torch.FloatTensor').to(device)
                output_batch = model(arr_out_gpu)
                output_batch = output_batch.detach().cpu().numpy()                
                output = np.append(output,output_batch,axis=0)
                pred_labels = np.argmax(output, axis=1)
                print("Predicted label: ",pred_labels)
                print("Predicted label: ",pred_labels[0])
                # pred_labels=[pred_labels]
                pred_all.append(pred_labels)

                if pred_labels[0]==0:
                    print("Class 0: Normal")
                    id_terms=parameters.cytomine_id_c0_term
                    pred_c0=pred_c0+1
                elif pred_labels[0]==1:
                    print("Class 1: Tumor")
                    id_terms=parameters.cytomine_id_c1_term
                    pred_c1=pred_c1+1
                
                cytomine_annotations = AnnotationCollection()
                annotation=roi_geometry
                
                cytomine_annotations.append(Annotation(location=annotation.wkt,#location=roi_geometry,
                                                    id_image=id_image,#conn.parameters.cytomine_id_image,
                                                    id_project=project.id,
                                                    id_terms=[id_terms]))
                print(".",end = '',flush=True)


                #Send Annotation Collection (for this ROI) to Cytomine server in one http request
                ca = cytomine_annotations.save()

                # except:
                #     print("An exception occurred. Proceed with next annotations")


            end_prediction_time=time.time()

            job.update(status=Job.RUNNING, progress=90, statusComment="Finalising Classification....")
            pred_all=[pred_c0, pred_c1]            
            print("pred_all:", pred_all)
            im_pred = np.argmax(pred_all)
            print("image prediction:", im_pred)
            pred_total=pred_c0+pred_c1
            print("pred_total:",pred_total)
            print("pred_tumor:",pred_c1)
            print("pred_normal:",pred_c0)
                  
            end_time=time.time()
            print("Execution time: ",end_time-start_time)
            print("Prediction time: ",end_prediction_time-start_prediction_time)

            f.write("\n")
            f.write("Image ID;Class Prediction;Class 0 (Normal);Class 1 (Tumor);Total Prediction;Execution Time;Prediction Time\n")
            f.write("{};{};{};{};{};{};{}\n".format(id_image,im_pred,pred_c0,pred_c1,pred_total,end_time-start_time,end_prediction_time-start_prediction_time))
            
        f.close()
        
        job.update(status=Job.RUNNING, progress=99, statusComment="Summarizing results...")
        job_data = JobData(job.id, "Generated File", "classification_results.csv").save()
        job_data.upload(output_path)

    finally:
        logging.info("Deleting folder %s", working_path)
        shutil.rmtree(working_path, ignore_errors=True)
        logging.debug("Leaving run()")


    job.update(status=Job.TERMINATED, progress=100, statusComment="Finished.") 

if __name__ == "__main__":
    logging.debug("Command: %s", sys.argv)

    with cytomine.CytomineJob.from_cli(sys.argv) as cyto_job:
        run(cyto_job, cyto_job.parameters)

