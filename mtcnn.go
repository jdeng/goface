package goface

import (
	"io/ioutil"
	//	"log"
	"math"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type MtcnnDetector struct {
	modelFile string
	graph     *tf.Graph
	session   *tf.Session

	minSize         float64
	scaleFactor     float64
	scoreThresholds []float32
}

func NewMtcnnDetector(modelFile string) (*MtcnnDetector, error) {
	det := &MtcnnDetector{modelFile: modelFile, minSize: 20.0, scaleFactor: 0.709, scoreThresholds: []float32{0.6, 0.7, 0.7}}
	model, err := ioutil.ReadFile(modelFile)
	if err != nil {
		return nil, err
	}

	graph := tf.NewGraph()
	if err := graph.Import(model, ""); err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}

	det.graph = graph
	det.session = session

	return det, nil
}

func (det *MtcnnDetector) Close() {
	if det.session != nil {
		det.session.Close()
		det.session = nil
	}
}

func (det *MtcnnDetector) Config(scaleFactor, minSize float64, scoreThresholds []float32) {
	if scaleFactor > 0 {
		det.scaleFactor = scaleFactor
	}
	if minSize > 0 {
		det.minSize = minSize
	}
	if scoreThresholds != nil {
		det.scoreThresholds = scoreThresholds
	}
}

func (det *MtcnnDetector) DetectFaces(tensor *tf.Tensor) ([][]float32, error) {
	session := det.session
	graph := det.graph

	var err error
	var total_bbox, total_reg [][]float32
	var total_score []float32

	h := float32(tensor.Shape()[1])
	w := float32(tensor.Shape()[2])
	scales := scales(float64(h), float64(w), det.scaleFactor, det.minSize)
	// log.Println("scales:", scales)

	// stage 1
	for _, scale := range scales {
		img, err := resizeImage(tensor, scale)
		if err != nil {
			return nil, err
		}
		output, err := session.Run(
			map[tf.Output]*tf.Tensor{
				graph.Operation("pnet/input").Output(0): img,
			},
			[]tf.Output{
				graph.Operation("pnet/conv4-2/BiasAdd").Output(0),
				graph.Operation("pnet/prob1").Output(0),
			},
			nil)
		if err != nil {
			return nil, err
		}

		// log.Println("pnet:", img.Shape(), "=>", output[0].Shape(), ",", output[1].Shape())

		out0, _ := transpose(output[0], []int32{0, 2, 1, 3})
		out1, _ := transpose(output[1], []int32{0, 2, 1, 3})

		xreg := out0.Value().([][][][]float32)[0]
		xscore := out1.Value().([][][][]float32)[0]

		bbox, reg, score := generateBbox(xscore, xreg, scale, det.scoreThresholds[0])
		if len(bbox) == 0 {
			continue
		}

		bbox, reg, score, err = nms(bbox, reg, score, 0.5)
		if len(bbox) > 0 {
			total_bbox = append(total_bbox, bbox...)
			total_reg = append(total_reg, reg...)
			total_score = append(total_score, score...)
		}
	}

	// log.Println("stage 1 bbox:", len(total_bbox))

	if len(total_bbox) == 0 {
		return nil, nil
	}

	total_bbox, total_reg, total_score, err = nms(total_bbox, total_reg, total_score, 0.7)
	// log.Println("stage 1 nms bbox:", len(total_bbox), err)

	if len(total_bbox) == 0 {
		return nil, nil
	}

	//calibrate & square
	for i, box := range total_bbox {
		total_bbox[i] = square(adjustBbox(box, total_reg[i]))
	}

	// stage 2
	imgs, err := cropResizeImage(tensor, normalizeBbox(total_bbox, w, h), []int32{24, 24}, true)
	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("rnet/input").Output(0): imgs,
		},
		[]tf.Output{
			graph.Operation("rnet/conv5-2/conv5-2").Output(0),
			graph.Operation("rnet/prob1").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}

	// log.Println("rnet:", imgs.Shape(), "=>", output[0].Shape(), ",", output[1].Shape())

	//filter
	reg := output[0].Value().([][]float32)
	score := output[1].Value().([][]float32)
	total_bbox, total_reg, total_score = filterBbox(total_bbox, reg, score, det.scoreThresholds[1])
	// log.Println("stage 2, filter bbox: ", len(total_bbox))

	if len(total_bbox) == 0 {
		return nil, nil
	}

	total_bbox, total_reg, total_score, err = nms(total_bbox, total_reg, total_score, 0.7)
	// log.Println("stage 2, nms bbox: ", len(total_bbox), err)

	if len(total_bbox) == 0 {
		return nil, nil
	}

	//calibrate, square
	for i, box := range total_bbox {
		total_bbox[i] = square(adjustBbox(box, total_reg[i]))
	}

	// stage 3
	imgs, err = cropResizeImage(tensor, normalizeBbox(total_bbox, w, h), []int32{48, 48}, true)
	output, err = session.Run(
		map[tf.Output]*tf.Tensor{
			graph.Operation("onet/input").Output(0): imgs,
		},
		[]tf.Output{
			graph.Operation("onet/conv6-2/conv6-2").Output(0),
			graph.Operation("onet/conv6-3/conv6-3").Output(0),
			graph.Operation("onet/prob1").Output(0),
		},
		nil)
	if err != nil {
		return nil, err
	}

	// log.Println("onet:", imgs.Shape(), "=>", output[0].Shape(), ",", output[1].Shape(), ",", output[2].Shape())

	reg = output[0].Value().([][]float32)
	score = output[2].Value().([][]float32)
	total_bbox, total_reg, total_score = filterBbox(total_bbox, reg, score, det.scoreThresholds[2])
	// log.Println("stage 3, filter bbox: ", len(total_bbox))

	if len(total_bbox) == 0 {
		return nil, nil
	}

	for i, box := range total_bbox {
		total_bbox[i] = adjustBbox(box, total_reg[i])
	}

	total_bbox, _, total_score, err = nms(total_bbox, total_reg, total_score, 0.7)
	// log.Println("stage 3, nms bbox: ", len(total_bbox), err)

	return total_bbox, nil
}

func adjustBbox(bbox []float32, reg []float32) []float32 {
	if len(bbox) == 4 && len(reg) == 4 {
		w := bbox[2] - bbox[0] + 1.0
		h := bbox[3] - bbox[1] + 1.0

		bbox[0] += reg[0] * w
		bbox[1] += reg[1] * h
		bbox[2] += reg[2] * w
		bbox[3] += reg[3] * h
	}
	return bbox
}

func square(bbox []float32) []float32 {
	if len(bbox) != 4 {
		return bbox
	}
	w := bbox[2] - bbox[0]
	h := bbox[3] - bbox[1]

	l := w
	if l < h {
		l = h
	}

	bbox[0] = bbox[0] + 0.5*w - 0.5*l
	bbox[1] = bbox[1] + 0.5*h - 0.5*l
	bbox[2] = bbox[0] + l
	bbox[3] = bbox[1] + l
	return bbox
}

func normalizeBbox(bbox [][]float32, w, h float32) [][]float32 {
	out := make([][]float32, len(bbox))
	for i, box := range bbox {
		ibox := make([]float32, 4)
		//NOTE: y1, x1, y2, x2
		ibox[0] = box[1] / h
		ibox[1] = box[0] / w
		ibox[2] = box[3] / h
		ibox[3] = box[2] / w
		out[i] = ibox
	}

	return out
}

func filterBbox(bbox, reg [][]float32, score [][]float32, threshold float32) (nbbox, nreg [][]float32, nscore []float32) {
	for i, x := range score {
		if x[1] > threshold {
			nbbox = append(nbbox, bbox[i])
			nreg = append(nreg, reg[i])
			nscore = append(nscore, x[1])
		}
	}
	return
}

func generateBbox(imap [][][]float32, reg [][][]float32, scale float64, threshold float32) (bbox, nreg [][]float32, score []float32) {
	const (
		Stride   = 2.0
		CellSize = 12.0
	)

	for i, x := range imap {
		for j, y := range x {
			if y[1] > threshold {
				n := []float32{float32(math.Floor((Stride*float64(j)+1.0)/scale + 0.5)),
					float32(math.Floor((Stride*float64(i)+1.0)/scale + 0.5)),
					float32(math.Floor((Stride*float64(j)+1.0+CellSize)/scale + 0.5)),
					float32(math.Floor((Stride*float64(i)+1.0+CellSize)/scale + 0.5)),
				}
				bbox = append(bbox, n)
				nreg = append(nreg, reg[i][j])
				score = append(score, y[1])
			}
		}
	}

	return
}

func nms(bbox, reg [][]float32, score []float32, threshold float32) (nbbox, nreg [][]float32, nscore []float32, err error) {
	tbbox, _ := tf.NewTensor(bbox)
	tscore, _ := tf.NewTensor(score)

	s := op.NewScope()
	pbbox := op.Placeholder(s.SubScope("bbox"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1, 4)))
	pscore := op.Placeholder(s.SubScope("score"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1)))

	out := op.NonMaxSuppression(s, pbbox, pscore, op.Const(s.SubScope("max_len"), int32(len(bbox))), op.NonMaxSuppressionIouThreshold(threshold))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pbbox: tbbox, pscore: tscore}, []tf.Output{out})
	if err != nil {
		return
	}

	pick := outs[0]
	if pick != nil {
		if idx, ok := pick.Value().([]int32); ok {
			for _, i := range idx {
				nbbox = append(nbbox, bbox[i])
				nreg = append(nreg, reg[i])
				nscore = append(nscore, score[i])
			}
		}
	}

	return
}

func CropResizeImage(img *tf.Tensor, bbox [][]float32, size []int32) (*tf.Tensor, error) {
	h := float32(img.Shape()[1])
	w := float32(img.Shape()[2])
	return cropResizeImage(img, normalizeBbox(bbox, w, h), size, false)
}

func cropResizeImage(img *tf.Tensor, bbox [][]float32, size []int32, normalize bool) (*tf.Tensor, error) {
	tbbox, _ := tf.NewTensor(bbox)

	s := op.NewScope()
	pimg := op.Placeholder(s.SubScope("img"), tf.Float, op.PlaceholderShape(tf.MakeShape(1, -1, -1, 3)))
	pbbox := op.Placeholder(s.SubScope("bbox"), tf.Float, op.PlaceholderShape(tf.MakeShape(-1, 4)))
	ibidx := op.Const(s.SubScope("bidx"), make([]int32, len(bbox)))
	isize := op.Const(s.SubScope("size"), size)

	//	log.Println("cropResize", img.Shape(), ",", tbbox.Shape())

	out := op.CropAndResize(s, pimg, pbbox, ibidx, isize)
	if normalize {
		out = normalizeImage(s, out)
	}

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pimg: img, pbbox: tbbox}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func resizeImage(img *tf.Tensor, scale float64) (*tf.Tensor, error) {
	h := int32(math.Ceil(float64(img.Shape()[1]) * scale))
	w := int32(math.Ceil(float64(img.Shape()[2]) * scale))

	s := op.NewScope()
	pimg := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(1, -1, -1, 3)))

	out := op.ResizeBilinear(s, pimg, op.Const(s.SubScope("size"), []int32{h, w}))
	out = normalizeImage(s, out)

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pimg: img}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func normalizeImage(s *op.Scope, input tf.Output) tf.Output {
	out := op.Mul(s, op.Sub(s, input, op.Const(s.SubScope("mean"), float32(127.5))),
		op.Const(s.SubScope("scale"), float32(0.0078125)))
	out = op.Transpose(s, out, op.Const(s.SubScope("perm"), []int32{0, 2, 1, 3}))
	return out
}

func transpose(img *tf.Tensor, perm []int32) (*tf.Tensor, error) {
	s := op.NewScope()
	in := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(-1, -1, -1, -1)))
	out := op.Transpose(s, in, op.Const(s, perm))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{in: img}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func runScope(s *op.Scope, inputs map[tf.Output]*tf.Tensor, outputs []tf.Output) ([]*tf.Tensor, error) {
	graph, err := s.Finalize()
	if err != nil {
		return nil, err
	}

	session, err := tf.NewSession(graph, nil)
	if err != nil {
		return nil, err
	}
	defer session.Close()
	return session.Run(inputs, outputs, nil)
}

func scales(h, w float64, factor, minSize float64) []float64 {
	minl := h
	if minl > w {
		minl = w
	}

	m := 12.0 / minSize
	minl = minl * m

	var scales []float64
	for count := 0; minl > 12.0; {
		scales = append(scales, m*math.Pow(factor, float64(count)))
		minl = minl * factor
		count += 1
	}

	return scales
}

func TensorFromJpeg(bytes []byte) (*tf.Tensor, error) {
	tensor, err := tf.NewTensor(string(bytes))
	if err != nil {
		return nil, err
	}

	s := op.NewScope()
	input := op.Placeholder(s, tf.String)
	out := op.ExpandDims(s,
		op.Cast(s, op.DecodeJpeg(s, input, op.DecodeJpegChannels(3)), tf.Float),
		op.Const(s.SubScope("make_batch"), int32(0)))

	outs, err := runScope(s, map[tf.Output]*tf.Tensor{input: tensor}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}
