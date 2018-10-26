let sketch = function (s) {

    s.renderEmoji = function () {
        const parameters = tf.tensor2d([s.parameters.map(p => p.value())]);
        tf.tidy(() => {
            s.model.outputLayers[0].predict(parameters).data()
                .then(arr => {
                    parameters.dispose();
                    let b = tf.scalar(0);
                    let a = tf.reshape(arr, [128, 128]);
                    let c = a.maximum(b);
                    a.dispose();
                    b.dispose();
                    tf.toPixels(c, s.canvas.canvas).then(() => c.dispose());
                });
        });
    };

    s.sliderChanged = async function () {
        for (let i = 0; i < s.parameters.length; i++) {
            s.inputs[i].value(s.parameters[i].value());
        }
        s.updatePermalink();
        s.renderEmoji();
    };

    s.updatePermalink = function () {
        s.permalink.html("");
        const url = new URL(document.location);
        for (let i = 0; i < s.inputs.length; i++) {
            url.searchParams.set(`v${i}`, s.parameters[i].value());
        }
        s.createA(url.toString(), url.toString()).parent(s.permalink);
    };

    s.newInput = function (event) {
        for (let i = 0; i < s.inputs.length; i++) {
            s.parameters[i].value(s.inputs[i].value());
        }
        s.updatePermalink();
        if (event && !isNaN(parseInt(event.target.value))) {
            s.renderEmoji();
        }
    };

    s.setup = async function () {
        s.canvas = s.createCanvas(128, 128);
        s.parameters = [];
        s.inputs = [];
        const url = new URL(document.location);
        for (let i = 0; i < 8; i++) {
            let div = s.createDiv();
            s.createSpan(`Variable ${i + 1}: `).parent(div);
            let value = parseFloat(url.searchParams.get(`v${i}`) || 0);
            s.parameters.push(s.createSlider(-3, 3, value, 0.0001).parent(div).input(s.sliderChanged).size(384));
            s.inputs.push(s.createInput(value.toString(), "number").parent(div).input(s.newInput).attribute("step", "0.0001"));
        }
        s.permalink = s.createDiv();

        s.model = await tf.loadModel("./model.json");
        s.newInput();
        s.renderEmoji();
    };
};

let myp5 = new p5(sketch);
