let sketch = function (s) {

    s.renderEmoji = function () {
        return tf.tidy(() => {
            s.model.outputLayers[0].predict(tf.tensor2d([s.parameters.map(p => p.value())])).data()
                .then(arr => {
                    let b = tf.scalar(0);
                    let a = tf.reshape(arr, [128, 128]).maximum(b);
                    tf.toPixels(a, s.canvas.canvas);
                });
        });
    };

    s.sliderChanged = async function () {
        for (let i = 0; i < s.parameters.length; i++) {
            s.inputs[i].value(s.parameters[i].value());
        }
        s.updatePermalink();
        await s.renderEmoji();
    };

    s.updatePermalink = function () {
        s.permalink.html("");
        const url = new URL(document.location);
        for (let i = 0; i < s.inputs.length; i++) {
            url.searchParams.set(`v${i}`, s.parameters[i].value());
        }
        s.createA(url.toString(), url.toString()).parent(s.permalink);
    };

    s.newInput = async function (event) {
        for (let i = 0; i < s.inputs.length; i++) {
            s.parameters[i].value(s.inputs[i].value());
        }
        s.updatePermalink();
        if (event && !isNaN(parseInt(event.target.value))) {
            await s.renderEmoji();
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
            s.parameters.push(s.createSlider(-1, 1, value, 0.00001).parent(div).input(s.sliderChanged).size(384));
            s.inputs.push(s.createInput(value.toString(), "number").parent(div).input(s.newInput).attribute("step", "0.00001"));
        }
        s.permalink = s.createDiv();

        s.model = await tf.loadModel("./model.json");
        await s.newInput();
        await s.renderEmoji();
    };
};

let myp5 = new p5(sketch);
