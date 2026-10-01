// =====================================================
// CARBONSCOPE JAVASCRIPT
// =====================================================


// -----------------------------------------------------
// HELPER
// -----------------------------------------------------

const $ = (selector) =>
    document.querySelector(selector);


// -----------------------------------------------------
// PREDICTION
// -----------------------------------------------------

const predictionForm =
    $("#predictionForm");


predictionForm.addEventListener(
    "submit",
    async (event) => {

        event.preventDefault();


        const data =
            Object.fromEntries(
                new FormData(event.target).entries()
            );


        try {

            const response =
                await fetch(
                    "/predict",
                    {
                        method: "POST",

                        headers: {
                            "Content-Type":
                                "application/json"
                        },

                        body:
                            JSON.stringify(data)
                    }
                );


            const result =
                await response.json();


            const predictions =
                result.predictions;


            $("#predictionResult").innerHTML = `

                <span class="muted">
                    Gradient Boosting estimate
                </span>

                <br>

                <strong>
                    ${Number(
                        predictions.gradient_boosting || 0
                    ).toLocaleString()}
                    kg CO₂e
                </strong>

                <br>

                <span class="muted">

                    Linear Regression:
                    ${Number(
                        predictions.linear_regression || 0
                    ).toLocaleString()}

                    ·

                    Random Forest:
                    ${Number(
                        predictions.random_forest || 0
                    ).toLocaleString()}

                </span>

            `;

        }

        catch (error) {

            console.error(
                "Prediction error:",
                error
            );

            $("#predictionResult").innerHTML = `

                <span class="muted">
                    Unable to generate prediction.
                </span>

            `;

        }

    }
);


// -----------------------------------------------------
// LOAD ANALYTICS
// -----------------------------------------------------

async function loadAnalytics() {

    try {

        const response =
            await fetch(
                "/api/analytics"
            );


        const analytics =
            await response.json();


        // ---------------------------------------------
        // FEATURE IMPORTANCE
        // ---------------------------------------------

        createFeatureImportanceChart(
            analytics.importance
        );


        // ---------------------------------------------
        // MODEL PERFORMANCE
        // ---------------------------------------------

        createModelPerformanceChart(
            analytics.metrics
        );


        // ---------------------------------------------
        // ACTUAL VS PREDICTED
        // ---------------------------------------------

        createActualPredictedChart(
            analytics.actual_predicted
        );


    }

    catch (error) {

        console.error(
            "Analytics loading error:",
            error
        );

    }

}


// -----------------------------------------------------
// FEATURE IMPORTANCE CHART
// -----------------------------------------------------

function createFeatureImportanceChart(
    importance
) {

    const canvas =
        document.getElementById(
            "featureImportanceChart"
        );


    if (!canvas) {
        return;
    }


    // Sort highest to lowest
    importance =
        [...importance].sort(
            (a, b) =>
                Math.abs(b.importance) -
                Math.abs(a.importance)
        );


    const labels =
        importance.map(
            item =>
                item.feature
                    .replaceAll("_", " ")
            );



    const values =
        importance.map(
            item =>
                Number(item.importance)
        );


    new Chart(
        canvas,
        {

            type: "bar",

            data: {

                labels: labels,

                datasets: [

                    {

                        label:
                            "Permutation Importance",

                        data:
                            values,

                        borderWidth: 0,

                        borderRadius: 7

                    }

                ]

            },


            options: {

                responsive: true,

                maintainAspectRatio: false,


                plugins: {

                    legend: {

                        display: false

                    },


                    tooltip: {

                        callbacks: {

                            label:
                                function(context) {

                                    return (
                                        " Importance: " +
                                        Number(
                                            context.raw
                                        ).toFixed(3)
                                    );

                                }

                        }

                    }

                },


                scales: {

                    x: {

                        grid: {

                            display: false

                        },

                        ticks: {

                            maxRotation: 45,

                            minRotation: 0

                        }

                    },


                    y: {

                        beginAtZero: true,

                        title: {

                            display: true,

                            text:
                                "Importance"

                        }

                    }

                }

            }

        }
    );

}


// -----------------------------------------------------
// MODEL PERFORMANCE CHART
// -----------------------------------------------------

function createModelPerformanceChart(
    metrics
) {

    const canvas =
        document.getElementById(
            "modelPerformanceChart"
        );


    if (!canvas) {
        return;
    }


    const labels =
        metrics.map(
            item =>
                item.model
        );


    const values =
        metrics.map(
            item =>
                Number(item.r2)
        );


    new Chart(
        canvas,
        {

            type: "bar",

            data: {

                labels: labels,

                datasets: [

                    {

                        label:
                            "R² Score",

                        data:
                            values,

                        borderWidth: 0,

                        borderRadius: 7

                    }

                ]

            },


            options: {

                responsive: true,

                maintainAspectRatio: false,


                plugins: {

                    legend: {

                        display: false

                    },


                    tooltip: {

                        callbacks: {

                            label:
                                function(context) {

                                    return (
                                        " R²: " +
                                        Number(
                                            context.raw
                                        ).toFixed(4)
                                    );

                                }

                        }

                    }

                },


                scales: {

                    x: {

                        grid: {

                            display: false

                        },

                        ticks: {

                            maxRotation: 45,

                            minRotation: 0

                        }

                    },


                    y: {

                        min: 0,

                        max: 1,

                        title: {

                            display: true,

                            text:
                                "R² Score"

                        }

                    }

                }

            }

        }
    );

}


// -----------------------------------------------------
// ACTUAL VS PREDICTED
// -----------------------------------------------------

function createActualPredictedChart(
    points
) {

    const canvas =
        document.getElementById(
            "actualPredictedChart"
        );


    if (!canvas) {
        return;
    }


    const scatterData =
        points.map(
            item => ({

                x:
                    Number(
                        item.actual
                    ),

                y:
                    Number(
                        item.predicted
                    )

            })
        );


    new Chart(
        canvas,
        {

            type: "scatter",

            data: {

                datasets: [

                    {

                        label:
                            "Predicted emissions",

                        data:
                            scatterData,

                        pointRadius: 4,

                        pointHoverRadius: 6,

                        borderWidth: 0

                    }

                ]

            },


            options: {

                responsive: true,

                maintainAspectRatio: false,


                plugins: {

                    legend: {

                        display: true

                    },


                    tooltip: {

                        callbacks: {

                            label:
                                function(context) {

                                    return [
                                        "Actual: " +
                                        Number(
                                            context.raw.x
                                        ).toFixed(2),

                                        "Predicted: " +
                                        Number(
                                            context.raw.y
                                        ).toFixed(2)
                                    ];

                                }

                        }

                    }

                },


                scales: {

                    x: {

                        title: {

                            display: true,

                            text:
                                "Actual CO₂ Emissions (kg)"

                        }

                    },


                    y: {

                        title: {

                            display: true,

                            text:
                                "Predicted CO₂ Emissions (kg)"

                        }

                    }

                }

            }

        }
    );

}


// -----------------------------------------------------
// UNSUPERVISED ML
// -----------------------------------------------------

async function loadUnsupervised() {

    try {

        const response =
            await fetch(
                "/api/unsupervised"
            );


        const data =
            await response.json();


        const container =
            $("#unsupervised");


        container.innerHTML =
            Object.entries(data)
                .map(
                    ([key, value]) => `

                        <div class="stat card">

                            <span class="muted">

                                ${key
                                    .replaceAll(
                                        "_",
                                        " "
                                    )}

                            </span>

                            <strong>

                                ${
                                    Array.isArray(value)

                                    ?

                                    value
                                        .map(
                                            n =>
                                                Number(n)
                                                    .toFixed(2)
                                        )
                                        .join(" / ")

                                    :

                                    value
                                }

                            </strong>

                        </div>

                    `
                )
                .join("");

    }

    catch (error) {

        console.error(
            "Unsupervised ML error:",
            error
        );

    }

}


// -----------------------------------------------------
// EVALUATION
// -----------------------------------------------------

async function loadEvaluation() {

    try {

        const response =
            await fetch(
                "/api/evaluation"
            );


        const data =
            await response.json();


        const container =
            $("#evaluation");


        container.innerHTML =
            Object.entries(data)
                .map(
                    ([key, value]) => `

                        <span class="pill">

                            ${key
                                .replaceAll(
                                    "_",
                                    " "
                                )}:

                            ${
                                typeof value === "number"

                                ?

                                value.toFixed(3)

                                :

                                value
                            }

                        </span>

                    `
                )
                .join(" ");

    }

    catch (error) {

        console.error(
            "Evaluation error:",
            error
        );

    }

}


// -----------------------------------------------------
// START APPLICATION
// -----------------------------------------------------

async function initializeDashboard() {

    await loadAnalytics();

    await loadUnsupervised();

    await loadEvaluation();

}


initializeDashboard();