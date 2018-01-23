# goface
Face Detector based on MTCNN, tensorflow and golang
Implementation based on https://github.com/davidsandberg/facenet 
Model file `mtcnn.pb` is converted from `facenet` too (see `scripts/convert.py`. You will need to add `facenet/src` to PYTHONPATH).

# Usage

```
	bs, err := ioutil.ReadFile(*imgFile)
	img, err := goface.TensorFromJpeg(bs)
	det, err := goface.NewMtcnnDetector("mtcnn.pb")
	bbox, err := det.DetectFaces(img) //[][]float32, i.e., [x1,y1,x2,y2],...
```
See `cmd/detect.go`.

# Notes

* Not exactly the same (e.g., nms/padding is depending on tensorflow implementation).
* Not fully tested. Performance could a little bit worse.
