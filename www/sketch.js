let sketch = function (s) {

    async function renderEmoji() {
        return tf.tidy(() => {
            s.model.outputLayers[0].predict(tf.tensor2d([s.parameters.map(p => p.value())])).data()
                .then(arr => {
                    let b = tf.scalar(0);
                    let a = tf.reshape(arr, [128, 128]).maximum(b);
                    tf.toPixels(a, s.canvas.canvas);
                });
        });
    }

    s.sliderChanged = async function () {
        for (let i = 0; i < s.parameters.length; i++) {
            s.inputs[i].value(s.parameters[i].value());
        }
        await renderEmoji();
    };

    s.newInput = async function (event) {
        for (let i = 0; i < s.inputs.length; i++) {
            s.parameters[i].value(s.inputs[i].value());
        }
        if (!isNaN(parseInt(event.target.value))) {
            await renderEmoji();
        }
    };

    s.setup = async function () {
        s.canvas = s.createCanvas(128, 128);
        s.parameters = [];
        s.inputs = [];
        for (let i = 1; i < 9; i++) {
            let div = s.createDiv();
            s.createSpan(`Variable ${i}: `).parent(div);
            s.parameters.push(s.createSlider(-1, 1, 0, 0.00001).parent(div).input(s.sliderChanged).size(384));
            s.inputs.push(s.createInput("0").parent(div).input(s.newInput));
        }

        s.model = await tf.loadModel("./model.json");
        await s.sliderChanged();
    };
};

let myp5 = new p5(sketch);
