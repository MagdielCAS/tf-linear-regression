let
    xs = [],
    ys = [],
    m, b;

const
    learningRate = 0.5,
    optimizer = tf.train.sgd(learningRate);

// p5.js 

setup = () => {
    createCanvas(400, 400);
    m = tf.variable(tf.scalar(Math.random(1)));
    b = tf.variable(tf.scalar(Math.random(1)));
}

mousePressed = () => {
    xs.push(normalizeOne(mouseX, width));
    ys.push(_normalizeOne(mouseY, height));
}

draw = () => {

    tf.tidy(() => {
        if (xs.length > 0) {
            //train
            optimizer.minimize(() => loss(predict(xs), tensorArr(ys)));
        }
    })

    background(0);

    stroke(255);
    strokeWeight(8);

    for (let i = 0; i < xs.length; i++) {
        let px = energize(xs[i], width);
        let py = _energize(ys[i], height);
        point(px, py);
    }
    if (xs.length > 0) {
        strokeWeight(2);
        let lineX = [0, 1];

        const yT = tf.tidy(() => predict(lineX));
        let lineY = yT.dataSync();
        yT.dispose();

        line(
            energize(lineX[0], width),
            _energize(lineY[0], height),
            energize(lineX[1], width),
            _energize(lineY[1], height));
    }
    //console.log(tf.memory().numTensors);
}

//TensorFlow
tensorArr = (arr) =>
    tf.tensor1d(arr)

predict = (val) =>
    tensorArr(val).mul(m).add(b);

loss = (pred, label) =>
    pred.sub(label).square().mean();

//Math helpers

normalizeOne = (val, max) =>
    map(val, 0, max, 0, 1);

_normalizeOne = (val, max) =>
    map(val, 0, max, 1, 0);

energize = (val, max) =>
    map(val, 0, 1, 0, max);

_energize = (val, max) =>
    map(val, 0, 1, max, 0);