# detection_web_app
1. !git clone https://github.com/tensorflow/models.git
2. cd models/research
3. protoc object_detection/protos/*.proto --python_out=.
4. git clone https://github.com/cocodataset/cocoapi.git
5. cd cocoapi/PythonAPI
6. make
7. cp -r pycocotools models/research
8. cd models/research
9. cp object_detection/packages/tf2/setup.py .
10. python -m pip install .
11. python object_detection/builders/model_builder_tf2_test.py