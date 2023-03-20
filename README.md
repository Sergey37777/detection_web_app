# detection_web_app

## Instalation
1. Create venv
2. Install requirements with pip
3. git clone https://github.com/tensorflow/models.git
4. cd models/research
5. protoc object_detection/protos/*.proto --python_out=.
6. git clone https://github.com/cocodataset/cocoapi.git
7. cd cocoapi/PythonAPI
8. make
9. cp -r pycocotools models/research
10. cd models/research
11. cp object_detection/packages/tf2/setup.py .
12. python -m pip install .
13. python object_detection/builders/model_builder_tf2_test.py
14. Create images folder