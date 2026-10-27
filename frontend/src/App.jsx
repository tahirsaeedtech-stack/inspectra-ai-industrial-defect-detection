import { useState } from "react";
import {
  Activity,
  UploadCloud,
  ShieldCheck,
  Cpu,
  Clock3,
  AlertTriangle,
  CheckCircle2,
} from "lucide-react";

const API_URL =
  import.meta.env.VITE_API_URL || "http://127.0.0.1:8000";

const DEFECT_LABELS = {
  crazing: "Crazing",
  inclusion: "Inclusion",
  patches: "Patches",
  pitted_surface: "Pitted Surface",
  "rolled-in_scale": "Rolled-in Scale",
  scratches: "Scratches",
};

function App() {
  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [gradcam, setGradcam] = useState(null);
  const [explaining, setExplaining] = useState(false);
  const handleFileChange = (event) => {
    const selectedFile = event.target.files?.[0];

    if (!selectedFile) return;

    const allowedTypes = [
      "image/jpeg",
      "image/png",
      "image/bmp",
      "image/x-ms-bmp",
    ];

    if (
      !allowedTypes.includes(selectedFile.type) &&
      !/\.(jpg|jpeg|png|bmp)$/i.test(selectedFile.name)
    ) {
      setError("Please upload a JPG, PNG, or BMP image.");
      return;
    }

    if (selectedFile.size > 10 * 1024 * 1024) {
      setError("File must be smaller than 10 MB.");
      return;
    }

    if (preview) URL.revokeObjectURL(preview);


    setFile(selectedFile);
    setPreview(URL.createObjectURL(selectedFile));
    setResult(null);
    setError("");
    setGradcam(null);
  };

  const handlePredict = async () => {
    if (!file) return;

    setLoading(true);
    setError("");
    setResult(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/predict`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const data = await response.json();
        throw new Error(
          typeof data.detail === "string"
            ? data.detail
            : "Prediction failed."
        );
      }

      const data = await response.json();

      if (!data || !data.prediction) {
        throw new Error("Invalid prediction response.");
      }

      setResult(data);
    } catch (err) {
      setError(err.message || "Unable to connect to the API.");
    } finally {
      setLoading(false);
    }
  };

  const sortedProbabilities = result
    ? Object.entries(result.probabilities).sort(
      (a, b) => b[1] - a[1]
    )
    : [];

  // ===== GRAD-CAM FUNCTION START =====

  const handleExplain = async () => {
    if (!file) return;

    setExplaining(true);
    setError("");
    setGradcam(null);

    try {
      const formData = new FormData();
      formData.append("file", file);

      const response = await fetch(`${API_URL}/explain`, {
        method: "POST",
        body: formData,
      });

      if (!response.ok) {
        const errorData = await response.json();

        throw new Error(
          typeof errorData.detail === "string"
            ? errorData.detail
            : "Grad-CAM generation failed."
        );
      }

      const data = await response.json();
      setGradcam(data.gradcam_image);

    } catch (err) {
      setError(err.message || "Unable to generate explanation.");

    } finally {
      setExplaining(false);
    }
  };

  // ===== GRAD-CAM FUNCTION END =====

  return (
    <div className="min-h-screen bg-[#080d19] text-slate-100">

      {/* Header */}
      <header className="border-b border-slate-800 bg-[#0d1424]">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-5">
          <div className="flex items-center gap-3">
            <div className="rounded-xl bg-cyan-500/15 p-3">
              <Activity className="text-cyan-400" size={26} />
            </div>

            <div>
              <h1 className="text-2xl font-bold tracking-tight">
                Inspectra <span className="text-cyan-400">AI</span>
              </h1>
              <p className="text-xs text-slate-400">
                Industrial Surface Defect Intelligence
              </p>
            </div>
          </div>

          <div className="flex items-center gap-2 rounded-full border border-emerald-500/30 bg-emerald-500/10 px-4 py-2 text-sm text-emerald-400">
            <span className="h-2 w-2 rounded-full bg-emerald-400" />
            AI Inference Dashboard
          </div>
        </div>
      </header>

      <main className="mx-auto max-w-7xl px-6 py-10">
        {/* Introduction */}
        <section className="mb-10">
          <div className="mb-3 inline-flex items-center gap-2 rounded-full border border-cyan-500/20 bg-cyan-500/10 px-3 py-1 text-xs text-cyan-300">
            <Cpu size={14} />
            Powered by EfficientNet-B0
          </div>

          <h2 className="text-3xl font-bold md:text-4xl">
            Surface Defect Detection
          </h2>

          <p className="mt-3 max-w-2xl text-slate-400">
            Upload a steel surface image to classify defects using
            a PyTorch deep learning model.
          </p>
        </section>

        {/* Feature Cards */}
        <section className="mb-8 grid gap-4 md:grid-cols-3">
          {[
            {
              icon: <ShieldCheck className="text-cyan-400" />,
              title: "6 Defect Categories",
              subtitle: "NEU surface defect classes",
            },
            {
              icon: <Cpu className="text-purple-400" />,
              title: "EfficientNet-B0",
              subtitle: "Transfer learning architecture",
            },
            {
              icon: <Clock3 className="text-emerald-400" />,
              title: "On-Demand Inference",
              subtitle: "FastAPI prediction service",
            },
          ].map((item) => (
            <div
              key={item.title}
              className="rounded-2xl border border-slate-800 bg-[#111a2b] p-5"
            >
              <div className="mb-3">{item.icon}</div>
              <h3 className="font-semibold">{item.title}</h3>
              <p className="mt-1 text-sm text-slate-400">
                {item.subtitle}
              </p>
            </div>
          ))}
        </section>

        {/* Main Dashboard */}
        <section className="grid gap-6 lg:grid-cols-2">
          {/* Upload Panel */}
          <div className="rounded-2xl border border-slate-800 bg-[#111a2b] p-6">
            <h3 className="mb-5 text-xl font-semibold">
              Image Inspection
            </h3>

            <label className="flex min-h-[300px] cursor-pointer flex-col items-center justify-center overflow-hidden rounded-xl border-2 border-dashed border-slate-700 bg-[#0b1220] transition hover:border-cyan-500">
              {preview ? (
                <img
                  src={preview}
                  alt="Uploaded steel surface"
                  className="max-h-[290px] max-w-full object-contain"
                />
              ) : (
                <div className="flex flex-col items-center px-6 text-center">
                  <UploadCloud
                    size={48}
                    className="mb-4 text-cyan-400"
                  />
                  <p className="font-medium">
                    Click to upload a steel surface image
                  </p>
                  <p className="mt-2 text-sm text-slate-500">
                    JPG, PNG, or BMP • Maximum 10 MB
                  </p>
                </div>
              )}

              <input
                type="file"
                accept=".jpg,.jpeg,.png,.bmp,image/jpeg,image/png,image/bmp"
                onChange={handleFileChange}
                className="hidden"
              />
            </label>

            {file && (
              <p className="mt-3 truncate text-sm text-slate-400">
                Selected: {file.name}
              </p>
            )}

            <button
              onClick={handlePredict}
              disabled={!file || loading}
              className="mt-5 w-full rounded-xl bg-cyan-500 px-6 py-3 font-semibold text-slate-950 transition hover:bg-cyan-400 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {loading ? "Analyzing Surface..." : "Analyze Image"}
            </button>

            {error && (
              <div className="mt-4 flex items-start gap-2 rounded-xl border border-red-500/30 bg-red-500/10 p-4 text-sm text-red-300">
                <AlertTriangle size={18} />
                <span>{error}</span>
              </div>
            )}
          </div>

          {/* Results Panel */}
          <div className="rounded-2xl border border-slate-800 bg-[#111a2b] p-6">
            <h3 className="mb-5 text-xl font-semibold">
              AI Inspection Results
            </h3>

            {!result ? (
              <div className="flex min-h-[300px] flex-col items-center justify-center text-center text-slate-500">
                <Activity size={48} className="mb-4 opacity-40" />
                <p>No inspection results yet</p>
                <p className="mt-2 text-sm">
                  Upload an image and run the analysis.
                </p>
              </div>
            ) : (
              <div>
                <div className="rounded-xl border border-cyan-500/20 bg-cyan-500/10 p-5">
                  <div className="flex items-center gap-2 text-cyan-300">
                    <CheckCircle2 size={18} />
                    <span className="text-sm">
                      Classification Complete
                    </span>
                  </div>

                  <h4 className="mt-3 text-3xl font-bold">
                    {DEFECT_LABELS[result.prediction] ||
                      result.prediction}
                  </h4>

                  <p className="mt-2 text-sm text-slate-400">
                    Model confidence score
                  </p>

                  <p className="mt-1 text-2xl font-semibold text-cyan-400">
                    {(result.confidence * 100).toFixed(2)}%
                  </p>
                </div>

                <div className="mt-6">
                  <h4 className="mb-4 font-semibold">
                    Class Probabilities
                  </h4>

                  <div className="space-y-4">
                    {sortedProbabilities.map(([name, probability]) => (
                      <div key={name}>
                        <div className="mb-2 flex justify-between text-sm">
                          <span>
                            {DEFECT_LABELS[name] || name}
                          </span>
                          <span className="text-slate-400">
                            {(probability * 100).toFixed(2)}%
                          </span>
                        </div>

                        <div className="h-2 overflow-hidden rounded-full bg-slate-800">
                          <div
                            className="h-full rounded-full bg-cyan-400"
                            style={{
                              width: `${probability * 100}%`,
                            }}
                          />
                        </div>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="mt-6 border-t border-slate-800 pt-5">
                  <div className="flex justify-between text-sm">
                    <span className="text-slate-400">
                      Inference Time
                    </span>
                    <span className="font-medium">
                      {result.processing_time_ms} ms
                    </span>
                  </div>

                  <div className="mt-3 flex justify-between text-sm">
                    <span className="text-slate-400">
                      Model
                    </span>
                    <span className="font-medium">
                      {result.model}
                    </span>
                  </div>
                </div>
              </div>
            )}
          </div>
        </section>

        {/* Grad-CAM Explainability */}
        <section className="mt-8 rounded-2xl border border-slate-800 bg-[#111a2b] p-6">

          <h3 className="text-xl font-semibold">
            Explainable AI — Grad-CAM
          </h3>

          <p className="mt-2 text-sm text-slate-400">
            Visualize the image regions influencing the model's prediction.
          </p>

          <button
            onClick={handleExplain}
            disabled={!result || explaining}
            className="mt-5 rounded-xl bg-purple-600 px-6 py-3 font-semibold text-white hover:bg-purple-500 disabled:opacity-50"
          >
            {explaining ? "Generating Heatmap..." : "Generate Grad-CAM"}
          </button>

          {gradcam && (
            <div className="mt-6">
              <h4 className="mb-3 font-medium">
                Model Attention Visualization
              </h4>

              <img
                src={gradcam}
                alt="Grad-CAM explanation"
                className="w-full max-w-md rounded-xl border border-slate-700"
              />

              <p className="mt-3 text-sm text-slate-400">
                Warmer regions indicate stronger positive contributions
                to the predicted class at the selected feature layer.
              </p>
            </div>
          )}

        </section>

        <p className="mt-8 text-center text-xs text-slate-500">
          Research prototype trained on the NEU surface defect dataset.
          Predictions require independent validation before industrial use.
        </p>
      </main>
    </div>
  );
}

export default App;