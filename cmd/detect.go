package main

import (
	"flag"
	"github.com/fogleman/gg"
	"github.com/jdeng/goface"
	"io/ioutil"
	"log"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
)

func main() {
	imgFile := flag.String("input", "1.jpg", "input jpeg file")
	outFile := flag.String("output", "1.png", "output png file")
	embedding := flag.Bool("embedding", false, "output embeddings")
	flag.Parse()

	bs, err := ioutil.ReadFile(*imgFile)
	if err != nil {
		log.Fatal(err)
	}

	img, err := goface.TensorFromJpeg(bs)
	if err != nil {
		log.Fatal(err)
	}

	det, err := goface.NewMtcnnDetector("mtcnn.pb")
	if err != nil {
		log.Fatal(err)
	}
	defer det.Close()

	// 0 for default
	det.Config(0, 0, []float32{0.7, 0.7, 0.95})

	bbox, err := det.DetectFaces(img)
	if err != nil {
		log.Fatal(err)
	}

	if len(bbox) == 0 {
		log.Println("No face found")
		return
	}

	var margin float32 = 16.0
	for _, box := range bbox {
		box[0] -= margin
		box[1] -= margin
		box[2] += margin
		box[3] += margin
	}

	log.Printf("%d faces found in %s\n", len(bbox), *imgFile)
	im, _ := gg.LoadImage(*imgFile)
	dc := gg.NewContextForImage(im)
	dc.SetRGBA(0, 0.8, 0.2, 0.4)
	for _, b := range bbox {
		dc.DrawRectangle(float64(b[0]), float64(b[1]), float64(b[2]-b[0]), float64(b[3]-b[1]))
	}
	dc.Fill()
	dc.SavePNG(*outFile)
	log.Printf("result saved to %s\n", *outFile)

	if *embedding {
		log.Printf("generating embddings for %d faces\n", len(bbox))
		fn, err := goface.NewFacenet("facenet.pb")
		if err != nil {
			log.Fatal(err)
		}
		defer fn.Close()

		var cropSize int32 = 160
		ximgs, err := goface.CropResizeImage(img, bbox, []int32{cropSize, cropSize})
		if err != nil {
			log.Fatal(err)
		}
		imgs := ximgs.Value().([][][][]float32)
		for _, img := range imgs {
			mean, std := goface.MeanStd(img)

			timg, err := tf.NewTensor([][][][]float32{img})
			if err != nil {
				log.Println(err)
				continue
			}

			wimg, err := goface.PrewhitenImage(timg, mean, std)
			if err != nil {
				log.Println(err)
				continue
			}

			emb, err := fn.Embedding(wimg)
			if err != nil {
				log.Println(err)
				continue
			}

			_ = emb
		}
	}
}
