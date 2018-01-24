# goface
Face detector/embeddings based on MTCNN, tensorflow and golang

Implementation based on https://github.com/davidsandberg/facenet . Tensorflow (1.4.1) and the golang binding are required. 

Model file `cmd/mtcnn.pb` is converted from `facenet` too (see `scripts/convert.py`. You will need to add `facenet/src` to PYTHONPATH to use it). You may need to regenerate the model file for a different version of tensorflow.

The `facenet` protobuf model file is available for download (see instructions from `facenet`).

# Usage

```
	// detection
	bs, err := ioutil.ReadFile(*imgFile)
	img, err := goface.TensorFromJpeg(bs)
	det, err := goface.NewMtcnnDetector("mtcnn.pb")
	bbox, err := det.DetectFaces(img) //[][]float32, i.e., [x1,y1,x2,y2],...

	// embeddings
	mean, std := goface.MeanStd(img)
	wimg, err := goface.PrewhitenImage(img, mean, std)
	fn, err := goface.NewFacenet("facenet.pb")
	emb, err := fn.Embedding(wimg)
```
See `cmd/detect.go`. Use `go build` to build the binary and run with `--help`.

# Notes

* Not exactly the same (e.g., nms/padding is depending on tensorflow implementation).
* Not fully tested. Performance could a little bit worse.
* Face landmark support not implemented.
