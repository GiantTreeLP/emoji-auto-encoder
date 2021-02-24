const decoder = function (s) {

    s.renderEmoji = function () {
        const parameters = tf.tensor2d([s.parameters.map(p => p.value())]);
        const prediction = s.model.outputLayers[0].predict(parameters);
        prediction.data()
            .then(arr => {
                parameters.dispose();
                const b = tf.scalar(0);
                const d = tf.scalar(1);
                const a = tf.reshape(arr, [128, 128, 1]);
                const c = a.maximum(b).minimum(d);
                a.dispose();
                b.dispose();
                tf.browser.toPixels(c, s.canvas.canvas).then(() => c.dispose());
                prediction.dispose();
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
            const div = s.createDiv();
            s.createSpan(`Variable ${i + 1}: `).parent(div);
            const value = parseFloat(url.searchParams.get(`v${i}`) || 0);
            s.parameters.push(s.createSlider(-1, 1, value, 0.0001).parent(div).input(s.sliderChanged).size(384));
            s.inputs.push(s.createInput(value.toString(), "number").parent(div).input(s.newInput).attribute("step", "0.0001"));
        }
        s.permalink = s.createDiv();

        s.model = await tf.loadLayersModel("./model.json");
        s.newInput();
        s.renderEmoji();
    };
};

new p5(decoder);


const denoiser = function (s) {
    s.fileloaded = async function (e) {
        s.loadImage(e.data, (img) => {
            s.inputImage.loadPixels();
            for (let y = 0; y < Math.min(s.inputImage.height, img.height); y++) {
                for (let x = 0; x < Math.min(s.inputImage.width, img.width); x++) {
                    s.inputImage.set(x, y, s.color(img.get(x, y)));
                }
            }
            s.inputImage.updatePixels();
            tf.tidy(() => {
                const parameter = tf.reshape(tf.browser.fromPixels(s.inputImage.canvas, 4), [-1, 128, 128, 1]).asType('float32')
                    .div(tf.scalar(255));
                const prediction = s.model.predict(parameter);
                prediction.data().then(arr => {
                    const b = tf.scalar(0);
                    const d = tf.scalar(1);
                    const a = tf.reshape(arr, [128, 128]);
                    const c = a.maximum(b).minimum(d);
                    tf.browser.toPixels(c, s.outputImage.canvas).then(() => {
                        a.dispose();
                        b.dispose();
                        c.dispose();
                    });
                });
            });
        });
    };

    s.setup = async function () {
        const div = s.createDiv();
        s.canvas = s.createCanvas(256, 128).parent(div);
        s.input = s.createFileInput(s.fileloaded).parent(div);
        s.inputImage = s.createImage(128, 128);
        s.outputImage = s.createImage(128, 128);
        s.model = await tf.loadLayersModel("./model.json");
    };

    s.draw = function () {
        s.image(s.inputImage, 0, 0, 128, 128);
        s.image(s.outputImage, 128, 0, 128, 128);
    };

};

new p5(denoiser);