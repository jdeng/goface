package main

import (
	"flag"
	"github.com/fogleman/gg"
	"github.com/jdeng/goface"
	"io/ioutil"
	"log"
)

func main() {
	det, err := goface.NewMtcnnDetector("mtcnn.pb")
	if err != nil {
		log.Fatal(err)
	}

	imgFile := flag.String("input", "1.jpg", "input jpeg file")
	outFile := flag.String("output", "1.png", "output png file")
	flag.Parse()

	bs, err := ioutil.ReadFile(*imgFile)
	if err != nil {
		log.Fatal(err)
	}

	img, err := goface.TensorFromJpeg(bs)
	if err != nil {
		log.Fatal(err)
	}

	bbox, err := det.DetectFaces(img)
	if err != nil {
		log.Fatal(err)
	}

	if len(bbox) == 0 {
		log.Println("No face found")
		return
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
}
