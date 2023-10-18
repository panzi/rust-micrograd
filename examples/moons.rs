use micrograd::{MLP, Module, Value, Number, Loss, cmp_number_ref};

fn main() {
    // Generated in Python with:
    // >>> from sklearn.datasets import make_moons
    // >>> X, y = make_moons(n_samples=100, noise=0.1)
    // >>> y = y*2 - 1

    let X = [
        [ 0.70782719,  0.63819789],
        [ 0.18015119,  0.8581158 ],
        [ 1.01410596, -0.45868371],
        [ 0.76986458, -0.44900259],
        [ 1.03773856,  0.23904158],
        [ 2.00780586,  0.1803901 ],
        [ 1.2052097 , -0.41031182],
        [ 0.52691413,  0.84160051],
        [-1.0242698 ,  0.37502457],
        [ 0.38339075,  1.0169598 ],
        [-0.0843429 ,  0.1327951 ],
        [-0.01802227, -0.2894898 ],
        [ 0.71842208,  0.40616139],
        [ 1.82429557,  0.00389606],
        [-0.66058386,  0.69849975],
        [ 0.68953281,  0.68565261],
        [-0.75009308,  0.85070821],
        [ 1.07681727,  0.25859468],
        [-0.80761846,  0.42928336],
        [ 0.07546165,  0.95506413],
        [-0.77061614,  0.67999391],
        [-0.75316293,  0.74327387],
        [ 0.15766837,  0.39766036],
        [ 1.50043415, -0.43082263],
        [ 0.39880061, -0.00575212],
        [ 0.81386785, -0.55923758],
        [-0.18103993,  1.08558569],
        [ 0.58854121,  0.51535079],
        [ 1.41151588, -0.55698455],
        [ 1.77229931, -0.41804515],
        [ 2.07964481,  0.11249148],
        [-0.35200954,  0.98919393],
        [ 1.47494889, -0.3774693 ],
        [ 1.92693575,  0.44422496],
        [ 1.99379656,  0.55291086],
        [ 0.90517099,  0.44773999],
        [ 0.07370635,  0.49522953],
        [ 0.02359471,  0.01002326],
        [-0.83989023,  0.85329983],
        [ 1.36830639, -0.48678748],
        [ 0.51625448, -0.29673798],
        [ 0.81208326,  0.30929297],
        [ 1.79427364,  0.08918423],
        [ 0.0958168 ,  1.07400113],
        [ 0.83463982, -0.62494117],
        [-1.11668064,  0.30005688],
        [-0.91629643,  0.31340072],
        [-0.14835282,  0.50554403],
        [ 1.14321049, -0.26588053],
        [ 1.04064165, -0.48794757],
        [ 0.6340955 ,  0.83915161],
        [ 0.76699831,  0.76188465],
        [-0.97274415, -0.06419751],
        [-0.15458669,  0.89874315],
        [-1.09810222, -0.0667073 ],
        [ 0.98882477,  0.16271646],
        [ 0.64472846,  0.70308145],
        [ 0.36844849,  0.98616532],
        [ 0.21533679,  0.11543269],
        [ 0.59262516, -0.30005244],
        [ 0.1815361 , -0.22461937],
        [-0.07610532,  0.36857433],
        [ 0.21618992, -0.14601164],
        [ 1.15545873, -0.45644428],
        [ 0.38792656, -0.43661694],
        [ 2.08324913,  0.32814873],
        [ 0.36136624,  0.75121216],
        [ 0.11576311,  0.91770334],
        [ 1.00720928,  0.18790394],
        [ 0.01919863,  0.96152525],
        [ 1.12517178, -0.21291343],
        [ 1.91507855,  0.40355582],
        [ 0.12554147,  0.20683176],
        [ 1.58465129, -0.37410464],
        [ 0.29872674, -0.33215256],
        [ 1.81178968, -0.20744855],
        [ 0.90881247,  0.02308503],
        [-0.40494888,  0.7891547 ],
        [ 0.21434566,  0.23127185],
        [ 0.92513802,  0.47746002],
        [-0.5069471 ,  0.84312124],
        [ 0.94750065, -0.23017591],
        [ 0.50938413, -0.34588786],
        [-1.06670522,  0.31545926],
        [-0.94071975,  0.44952784],
        [-0.0971809 ,  0.98333104],
        [ 0.48345934, -0.21134621],
        [ 0.32697403,  1.08087061],
        [ 1.80610526, -0.00897735],
        [-0.84814924,  0.32360855],
        [ 1.57600272, -0.34216431],
        [ 0.21676336,  1.02350696],
        [ 1.66862797, -0.31872215],
        [ 2.0207912 ,  0.13754996],
        [-0.2442827 ,  0.7251728 ],
        [-1.07102948,  0.12684189],
        [ 0.81528357,  0.2488677 ],
        [ 1.84376012,  0.03400494],
        [-0.54150896,  0.82534897],
        [ 0.3929311 , -0.23956422]
    ];

    let y = [
        -1, -1,  1,  1, -1,  1,  1, -1, -1, -1,  1,  1, -1,  1, -1, -1, -1,
        -1, -1, -1, -1, -1,  1,  1,  1,  1, -1, -1,  1,  1,  1, -1,  1,  1,
         1, -1,  1,  1, -1,  1,  1, -1,  1, -1,  1, -1, -1,  1,  1,  1, -1,
        -1, -1, -1, -1, -1, -1, -1,  1,  1,  1,  1,  1,  1,  1,  1, -1, -1,
        -1, -1,  1,  1,  1,  1,  1,  1, -1, -1,  1, -1, -1,  1,  1, -1, -1,
        -1,  1, -1,  1, -1,  1, -1,  1,  1, -1, -1, -1,  1, -1,  1
    ];

    let mut rng = rand::thread_rng();
    // 2-layer neural network
    let mut model = MLP::new(2, &[16, 16, 1], &mut rng);

    println!("{:#?}", model);
    println!();
    println!("number of parameters: {}", model.count_parameters());
    println!();

    let loss = |model: &MLP| {
        let inputs = X.map(|xrow| xrow.map(Value::new));

        // forward the model to get scores
        let scores: Vec<_> = inputs.iter().map(
            |input| model.forward(input).first().unwrap().clone()
        ).collect();

        // sum "max-margin" loss
        let losses: Vec<_> = y.iter().cloned().zip(scores.iter()).map(
            |(yi, scorei)| (scorei.clone() * Number::from(-yi) + 1.0).relu()
        ).collect();
        let data_loss = losses.iter().cloned().sum::<Value>() * (1.0 / losses.len() as Number);

        // L2 regularization
        let alpha: Number = 1e-4;
        let reg_loss = model.fold_paramters(Value::new(0.0), |acc, value| acc + value.clone()) * alpha;
        let total_loss = data_loss + reg_loss;

        // also get accuracy
        let mut sum_accuracy: usize = 0;
        for (yi, scorei) in y.iter().cloned().zip(scores) {
            sum_accuracy += ((yi > 0) == (scorei.value() > 0.0)) as usize;
        }

        let accuracy = sum_accuracy as Number / y.len() as Number;

        Loss {
            total: total_loss,
            accuracy
        }
    };

    println!("optimizing:");
    for (k, loss) in model.optimize(100, loss) {
        println!("step {} loss {}, accuracy {}%", k, loss.total.value(), loss.accuracy * 100.0);
    }

    // visualize decision boundary
    let h = 0.25;
    let x_min = X.iter().map(|row| row[0]).min_by(cmp_number_ref).unwrap_or(0.0);
    let x_max = X.iter().map(|row| row[0]).max_by(cmp_number_ref).unwrap_or(0.0);
    let y_min = X.iter().map(|row| row[1]).min_by(cmp_number_ref).unwrap_or(0.0);
    let y_max = X.iter().map(|row| row[1]).max_by(cmp_number_ref).unwrap_or(0.0);

    let mut x = x_min;
    let mut xmesh = Vec::new();
    while x < x_max {
        let mut y = y_min;
        while y < y_max {
            xmesh.push([x, y]);
            y += h;
        }
        x += h;
    }

    let scores: Vec<_> = xmesh.iter().map(|xrow| model.forward(&xrow.map(Value::new)).first().unwrap().clone()).collect();

    let z: Vec<_> = scores.iter().map(|value| value.value() > 0.0).collect();

    // TODO: somehow plot it

}

fn arange(start: Number, end: Number, step: Number) -> Vec<Number> {
    let mut xs = Vec::new();
    let mut x = start;

    while x < end {
        xs.push(x);
        x += step;
    }

    xs
}

fn meshgrid(xs: &[Number], ys: &[Number]) -> (Vec<Vec<Number>>, Vec<Vec<Number>>) {
    (
        (0..ys.len()).map(|_| Vec::from(xs)).collect(),
        ys.iter().cloned().map(|y| vec![y; xs.len()]).collect(),
    )
}
