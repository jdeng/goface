package goface

import (
	"io/ioutil"
	"math"

	tf "github.com/tensorflow/tensorflow/tensorflow/go"
	"github.com/tensorflow/tensorflow/tensorflow/go/op"
)

type Facenet struct {
	modelFile string
	graph     *tf.Graph
	session   *tf.Session
}

func NewFacenet(modelFile string) (*Facenet, error) {
	fn := &Facenet{modelFile: modelFile}
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

	fn.graph = graph
	fn.session = session

	return fn, nil
}

func (fn *Facenet) Close() {
	if fn.session != nil {
		fn.session.Close()
		fn.session = nil
	}
}

func MeanStd(img [][][]float32) (mean float32, std float32) {
	count := len(img) * len(img[0]) * len(img[0][0])
	for _, x := range img {
		for _, y := range x {
			for _, z := range y {
				mean += z
			}
		}
	}
	mean /= float32(count)

	for _, x := range img {
		for _, y := range x {
			for _, z := range y {
				std += (z - mean) * (z - mean)
			}
		}
	}

	xstd := math.Sqrt(float64(std) / float64(count-1))
	minstd := 1.0 / math.Sqrt(float64(count))
	if xstd < minstd {
		xstd = minstd
	}

	std = float32(xstd)
	return
}

func PrewhitenImage(img *tf.Tensor, mean, std float32) (*tf.Tensor, error) {
	s := op.NewScope()
	pimg := op.Placeholder(s, tf.Float, op.PlaceholderShape(tf.MakeShape(1, -1, -1, 3)))

	out := op.Mul(s, op.Sub(s, pimg, op.Const(s.SubScope("mean"), mean)),
		op.Const(s.SubScope("scale"), float32(1.0)/std))
	outs, err := runScope(s, map[tf.Output]*tf.Tensor{pimg: img}, []tf.Output{out})
	if err != nil {
		return nil, err
	}

	return outs[0], nil
}

func (fn *Facenet) Embedding(img *tf.Tensor) ([][]float32, error) {
	session := fn.session
	graph := fn.graph

	input := graph.Operation("input").Output(0)
	emb := graph.Operation("embeddings").Output(0)
	phase_train := graph.Operation("phase_train").Output(0)

	t, _ := tf.NewTensor(bool(false))

	output, err := session.Run(
		map[tf.Output]*tf.Tensor{
			input:       img,
			phase_train: t,
		},
		[]tf.Output{
			emb,
		},
		nil)
	if err != nil {
		return nil, err
	}

	out := output[0].Value().([][]float32)
	return out, nil
}
