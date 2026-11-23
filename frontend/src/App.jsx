import { useEffect, useMemo, useRef, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  AreaChart,
  Area,
} from "recharts";

const BACKEND_URL = "http://127.0.0.1:5000";

const EXERCISES = {
  ABD: "Abduction",
  FEL: "Forward Elevation",
  IR: "Internal Rotation",
  PEN: "Pendulum",
  REST: "Rest",
  WRONG: "Wrong Movement",
};

const EXERCISE_AXIS = {
  ABD: "angle_x",
  FEL: "angle_y",
  IR: "angle_z",
  PEN: "angle_x",
};

const SENSOR_KEYS = [
  "ax",
  "ay",
  "az",
  "wx",
  "wy",
  "wz",
  "angle_x",
  "angle_y",
  "angle_z",
];

const QUALITY_COLORS = ["#16a34a", "#f59e0b", "#ef4444"];

function App() {
  const [backend, setBackend] = useState({
    status: "Checking backend...",
    sensor_status: "Not started",
    sensor_connected: false,
    sensor_running: false,
    model_loaded: false,
  });

  const [reading, setReading] = useState({
    ax: 0,
    ay: 0,
    az: 0,
    wx: 0,
    wy: 0,
    wz: 0,
    angle_x: 0,
    angle_y: 0,
    angle_z: 0,
  });

  const [samples, setSamples] = useState(0);

  const [prediction, setPrediction] = useState({
    exercise: "Waiting",
    confidence: 0,
    feedback: "Start the WIT sensor to begin.",
  });

  const predictionRef = useRef({
    exercise: "Waiting",
    confidence: 0,
    feedback: "Start the WIT sensor to begin.",
  });

  const [liveData, setLiveData] = useState([]);
  const liveDataRef = useRef([]);

  const [demoMode, setDemoMode] = useState(false);
  const demoTimerRef = useRef(null);

  const [selectedExercise, setSelectedExercise] = useState("ABD");
  const selectedExerciseRef = useRef("ABD");

  const [isCalibrating, setIsCalibrating] = useState(false);
  const [isActualSession, setIsActualSession] = useState(false);

  const isCalibratingRef = useRef(false);
  const isActualSessionRef = useRef(false);

  const [calibrationData, setCalibrationData] = useState([]);
  const calibrationDataRef = useRef([]);

  const [baseline, setBaseline] = useState(null);
  const baselineRef = useRef(null);

  const [mainFeedback, setMainFeedback] = useState(
    "Start the sensor, calibrate 2–3 correct repetitions, then begin the actual exercise session."
  );

  const [repCount, setRepCount] = useState(0);
  const repCountRef = useRef(0);

  const [correctReps, setCorrectReps] = useState(0);
  const [needsReps, setNeedsReps] = useState(0);
  const [incorrectReps, setIncorrectReps] = useState(0);

  const [repHistory, setRepHistory] = useState([]);

  const repStateRef = useRef("down");
  const lastRepTimeRef = useRef(0);
  const currentRepSamplesRef = useRef([]);
  const pendulumLastSideRef = useRef(null);

  useEffect(() => {
    selectedExerciseRef.current = selectedExercise;
  }, [selectedExercise]);

  useEffect(() => {
    baselineRef.current = baseline;
  }, [baseline]);

  useEffect(() => {
    checkBackend();

    const latestTimer = setInterval(() => {
      if (!demoMode) {
        fetchLatest();
      }
    }, 250);

    const predictionTimer = setInterval(() => {
      if (!demoMode) {
        fetchPrediction();
      }
    }, 1200);

    return () => {
      clearInterval(latestTimer);
      clearInterval(predictionTimer);

      if (demoTimerRef.current) {
        clearInterval(demoTimerRef.current);
      }
    };
  }, [demoMode]);

  function scrollToSection(id) {
    const section = document.getElementById(id);

    if (section) {
      section.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }

  async function checkBackend() {
    try {
      const res = await fetch(`${BACKEND_URL}/health`);
      const data = await res.json();

      setBackend({
        status: "Backend connected",
        sensor_status: data.sensor_status || "Unknown",
        sensor_connected: data.sensor_connected || false,
        sensor_running: data.sensor_running || false,
        model_loaded: data.model_loaded || false,
      });
    } catch {
      setBackend({
        status: "Backend not connected",
        sensor_status: "Backend is not running",
        sensor_connected: false,
        sensor_running: false,
        model_loaded: false,
      });
    }
  }

  async function startRealSensor() {
    try {
      const res = await fetch(`${BACKEND_URL}/start_sensor`, {
        method: "POST",
      });

      const data = await res.json();

      setBackend((prev) => ({
        ...prev,
        sensor_status: data.sensor_status || "Sensor started",
      }));

      setMainFeedback(
        "Real WIT sensor started. Wait until the model window reaches 100 samples."
      );
    } catch {
      setBackend((prev) => ({
        ...prev,
        sensor_status: "Could not start sensor. Check backend.",
      }));

      setMainFeedback(
        "Could not start the real sensor. Check that the backend is running and WITMotion is closed."
      );
    }
  }

  async function stopRealSensor() {
    try {
      const res = await fetch(`${BACKEND_URL}/stop_sensor`, {
        method: "POST",
      });

      const data = await res.json();

      setBackend((prev) => ({
        ...prev,
        sensor_status: data.sensor_status || "Stopping sensor",
      }));

      stopActualSession();
      setMainFeedback("Real sensor stopped.");
    } catch {
      setBackend((prev) => ({
        ...prev,
        sensor_status: "Could not stop sensor.",
      }));
    }
  }

  async function fetchLatest() {
    try {
      const res = await fetch(`${BACKEND_URL}/latest`);
      const data = await res.json();

      if (data.reading) {
        const clean = normalizeReading(data.reading);
        pushReading(clean);
      }

      setSamples(data.samples || 0);

      setBackend((prev) => ({
        ...prev,
        sensor_status: data.sensor_status || prev.sensor_status,
        sensor_connected: data.sensor_connected || false,
        sensor_running: data.sensor_running || false,
      }));
    } catch {
      setBackend((prev) => ({
        ...prev,
        status: "Backend disconnected",
      }));
    }
  }

  async function fetchPrediction() {
    try {
      const res = await fetch(`${BACKEND_URL}/predict`);
      const data = await res.json();

      const newPrediction = {
        exercise: data.exercise || "Waiting",
        confidence: Number(data.confidence || 0),
        feedback: data.feedback || "Waiting for prediction.",
      };

      setPrediction(newPrediction);
      predictionRef.current = newPrediction;
    } catch {
      const errorPrediction = {
        exercise: "Backend Error",
        confidence: 0,
        feedback: "Could not get prediction from backend.",
      };

      setPrediction(errorPrediction);
      predictionRef.current = errorPrediction;
    }
  }

  function normalizeReading(input) {
    return {
      time: input.time || Date.now() / 1000,
      ax: Number(input.ax || 0),
      ay: Number(input.ay || 0),
      az: Number(input.az || 0),
      wx: Number(input.wx || 0),
      wy: Number(input.wy || 0),
      wz: Number(input.wz || 0),
      angle_x: Number(input.angle_x || input.angleX || 0),
      angle_y: Number(input.angle_y || input.angleY || 0),
      angle_z: Number(input.angle_z || input.angleZ || 0),
    };
  }

  function pushReading(newReading) {
    setReading(newReading);

    const previous = liveDataRef.current;

    const newPoint = {
      ...newReading,
      index: previous.length > 0 ? previous[previous.length - 1].index + 1 : 1,
      acc_mag: Math.sqrt(
        newReading.ax ** 2 + newReading.ay ** 2 + newReading.az ** 2
      ),
      gyro_mag: Math.sqrt(
        newReading.wx ** 2 + newReading.wy ** 2 + newReading.wz ** 2
      ),
    };

    liveDataRef.current = [...previous, newPoint].slice(-140);
    setLiveData(liveDataRef.current);

    if (isCalibratingRef.current) {
      calibrationDataRef.current = [...calibrationDataRef.current, newReading];
      setCalibrationData(calibrationDataRef.current);
    }

    if (isActualSessionRef.current) {
      detectRep(newReading);
    }
  }

  function startDemoMode() {
    setDemoMode(true);

    if (demoTimerRef.current) {
      clearInterval(demoTimerRef.current);
    }

    let t = 0;

    demoTimerRef.current = setInterval(() => {
      t += 0.1;

      const exercise = selectedExerciseRef.current;

      let angle_x = 15 + Math.abs(Math.sin(t)) * 75;
      let angle_y = 8 + Math.abs(Math.sin(t * 0.7)) * 35;
      let angle_z = 5 + Math.abs(Math.sin(t * 0.4)) * 22;

      if (exercise === "FEL") {
        angle_x = 12 + Math.abs(Math.sin(t * 0.7)) * 30;
        angle_y = 10 + Math.abs(Math.sin(t)) * 80;
      }

      if (exercise === "IR") {
        angle_x = 8 + Math.abs(Math.sin(t * 0.7)) * 22;
        angle_z = 5 + Math.abs(Math.sin(t)) * 75;
      }

      if (exercise === "PEN") {
        angle_x = Math.sin(t * 2.2) * 22;
        angle_y = Math.cos(t * 1.8) * 15;
        angle_z = Math.sin(t * 1.4) * 12;
      }

      const demoReading = {
        time: Date.now() / 1000,
        ax: Math.sin(t) * 0.6,
        ay: Math.cos(t) * 0.4,
        az: 9.7 + Math.sin(t * 0.7) * 0.2,
        wx: Math.sin(t * 2) * 30,
        wy: Math.cos(t * 1.5) * 22,
        wz: Math.sin(t * 1.3) * 18,
        angle_x,
        angle_y,
        angle_z,
      };

      pushReading(demoReading);

      const fakeConfidence = 86 + Math.sin(t) * 9;

      setPrediction({
        exercise,
        confidence: Number(fakeConfidence.toFixed(1)),
        feedback: `${EXERCISES[exercise]} movement pattern detected.`,
      });

      predictionRef.current = {
        exercise,
        confidence: Number(fakeConfidence.toFixed(1)),
        feedback: `${EXERCISES[exercise]} movement pattern detected.`,
      };

      setSamples(100);
    }, 80);

    setBackend((prev) => ({
      ...prev,
      sensor_status: "Demo mode running",
      sensor_connected: true,
      sensor_running: true,
    }));

    setMainFeedback(
      "Demo mode is running. Use this only as backup if live Bluetooth is unstable."
    );
  }

  function stopDemoMode() {
    setDemoMode(false);

    if (demoTimerRef.current) {
      clearInterval(demoTimerRef.current);
      demoTimerRef.current = null;
    }

    setBackend((prev) => ({
      ...prev,
      sensor_status: "Demo mode stopped",
      sensor_connected: false,
      sensor_running: false,
    }));

    stopActualSession();

    setMainFeedback("Demo mode stopped.");
  }

  function extractMovementFeatures(data) {
    if (!data || data.length === 0) return null;

    const features = {
      axisFeatures: {},
      vector: [],
      samples: data.length,
    };

    SENSOR_KEYS.forEach((key) => {
      const values = data.map((d) => Number(d[key] || 0));

      const min = Math.min(...values);
      const max = Math.max(...values);
      const range = max - min;
      const mean = values.reduce((sum, v) => sum + v, 0) / values.length;

      const variance =
        values.reduce((sum, v) => sum + Math.pow(v - mean, 2), 0) /
        values.length;

      const std = Math.sqrt(variance);

      const rms = Math.sqrt(
        values.reduce((sum, v) => sum + v * v, 0) / values.length
      );

      features.axisFeatures[key] = {
        min,
        max,
        range,
        mean,
        std,
        rms,
      };

      features.vector.push(mean);
      features.vector.push(range);
      features.vector.push(std);
      features.vector.push(rms);
    });

    const angleRanges = {
      angle_x: features.axisFeatures.angle_x.range,
      angle_y: features.axisFeatures.angle_y.range,
      angle_z: features.axisFeatures.angle_z.range,
    };

    const dominantAxis = Object.keys(angleRanges).reduce((a, b) =>
      angleRanges[a] > angleRanges[b] ? a : b
    );

    features.dominantAxis = dominantAxis;
    features.dominantRange = angleRanges[dominantAxis];

    features.rangeX = features.axisFeatures.angle_x.range;
    features.rangeY = features.axisFeatures.angle_y.range;
    features.rangeZ = features.axisFeatures.angle_z.range;

    return features;
  }

  function compareWithBaseline(
    current,
    base,
    exerciseName = selectedExerciseRef.current
  ) {
    if (!current || !base) {
      return {
        status: "No Baseline",
        similarity: 0,
        message: "Please record correct calibration reps first.",
      };
    }

    let totalScore = 0;
    let totalWeight = 0;

    SENSOR_KEYS.forEach((key) => {
      const baseAxis = base.axisFeatures[key];
      const currentAxis = current.axisFeatures[key];

      if (!baseAxis || !currentAxis) return;

      let weight = 1;

      if (key.includes("angle")) {
        weight = 4;
      } else if (key.startsWith("w")) {
        weight = exerciseName === "IR" ? 4 : 2.5;
      } else {
        weight = exerciseName === "PEN" ? 1 : 1.2;
      }

      const baseRange = Math.max(Math.abs(baseAxis.range), 1);
      const baseStd = Math.max(Math.abs(baseAxis.std), 1);
      const baseRms = Math.max(Math.abs(baseAxis.rms || 0), 1);

      const rangeDiff = Math.abs(currentAxis.range - baseAxis.range) / baseRange;
      const stdDiff = Math.abs(currentAxis.std - baseAxis.std) / baseStd;
      const rmsDiff =
        Math.abs((currentAxis.rms || 0) - (baseAxis.rms || 0)) / baseRms;

      let meanDiff = 0;

      if (key.includes("angle")) {
        const baseMean = Math.max(Math.abs(baseAxis.mean), 5);
        meanDiff = Math.abs(currentAxis.mean - baseAxis.mean) / baseMean;
      }

      const clippedRangeDiff = Math.min(rangeDiff, 1.3);
      const clippedStdDiff = Math.min(stdDiff, 1.3);
      const clippedRmsDiff = Math.min(rmsDiff, 1.3);
      const clippedMeanDiff = Math.min(meanDiff, 1.1);

      let axisError;

      if (exerciseName === "IR") {
        axisError =
          clippedRangeDiff * 0.5 +
          clippedStdDiff * 0.2 +
          clippedRmsDiff * 0.2 +
          clippedMeanDiff * 0.1;
      } else if (exerciseName === "PEN") {
        axisError =
          clippedRangeDiff * 0.72 +
          clippedStdDiff * 0.16 +
          clippedRmsDiff * 0.08 +
          clippedMeanDiff * 0.04;
      } else {
        axisError =
          clippedRangeDiff * 0.68 +
          clippedStdDiff * 0.16 +
          clippedRmsDiff * 0.1 +
          clippedMeanDiff * 0.06;
      }

      let axisScore = 100 - axisError * 100;
      axisScore = Math.max(0, Math.min(100, axisScore));

      totalScore += axisScore * weight;
      totalWeight += weight;
    });

    let similarity = totalWeight === 0 ? 0 : totalScore / totalWeight;

    const baseRange = Math.max(base.dominantRange, 1);
    const currentRange = Math.max(current.dominantRange, 1);
    const romRatio = currentRange / baseRange;

    const currentX = current.rangeX || 0;
    const currentY = current.rangeY || 0;
    const currentZ = current.rangeZ || 0;

    if (exerciseName === "ABD") {
      if (romRatio < 0.65) similarity -= 25;
      if (romRatio > 1.6) similarity -= 15;

      if (currentY < base.rangeY * 0.55 && currentX > base.rangeX * 0.85) {
        similarity -= 18;
      }

      if (current.dominantAxis !== base.dominantAxis) {
        similarity -= 10;
      }
    }

    if (exerciseName === "FEL") {
      if (romRatio < 0.65) similarity -= 25;
      if (romRatio > 1.6) similarity -= 15;

      if (current.dominantAxis !== base.dominantAxis) {
        similarity -= 10;
      }
    }

    if (exerciseName === "IR") {
      if (romRatio < 0.6 || romRatio > 1.55) {
        similarity -= 28;
      }

      if (current.dominantAxis !== base.dominantAxis) {
        similarity -= 16;
      }

      if (currentZ < base.rangeZ * 0.55) {
        similarity -= 20;
      }
    }

    if (exerciseName === "PEN") {
      if (romRatio >= 0.45 && romRatio <= 1.8) {
        similarity += 8;
      } else if (romRatio < 0.3) {
        similarity -= 15;
      }

      if (current.dominantAxis !== base.dominantAxis) {
        similarity -= 4;
      }
    }

    similarity = Math.max(0, Math.min(100, similarity));

    let correctThreshold = 70;
    let needsThreshold = 50;

    if (exerciseName === "IR") {
      correctThreshold = 76;
      needsThreshold = 58;
    }

    if (exerciseName === "PEN") {
      correctThreshold = 58;
      needsThreshold = 38;
    }

    if (similarity >= correctThreshold) {
      return {
        status: "Correct",
        similarity: similarity.toFixed(1),
        message:
          exerciseName === "IR"
            ? "Good internal rotation. The rotational pattern matches the calibrated baseline."
            : "Good movement. The rep matches the calibrated movement pattern.",
      };
    }

    if (similarity >= needsThreshold) {
      return {
        status: "Needs Improvement",
        similarity: similarity.toFixed(1),
        message:
          "The movement is close, but range, speed, or direction can be improved.",
      };
    }

    return {
      status: "Incorrect",
      similarity: similarity.toFixed(1),
      message:
        "Incorrect movement. The rep is too different from the calibrated exercise pattern.",
    };
  }

  function finishRepEvaluation(repFeatures, exerciseName) {
    const base = baselineRef.current;

    if (!base || !repFeatures) return;

    let result = compareWithBaseline(repFeatures, base, exerciseName);

    const livePred = String(predictionRef.current.exercise || "").toUpperCase();
    const liveConf = Number(predictionRef.current.confidence || 0);

    const allowedWaiting = ["WAITING", "COLLECTING", "BACKEND ERROR"];

    const modelIsUseful =
      livePred && !allowedWaiting.includes(livePred) && liveConf >= 55;

    if (modelIsUseful) {
      if (livePred === "WRONG") {
        result = {
          status: "Incorrect",
          similarity: Math.min(Number(result.similarity || 0), 35).toFixed(1),
          message:
            "Incorrect movement detected by the live ML model. Adjust posture and repeat slowly.",
        };
      } else if (livePred === "REST") {
        result = {
          status: "Incorrect",
          similarity: Math.min(Number(result.similarity || 0), 35).toFixed(1),
          message:
            "Rest or no clear exercise movement was detected. Repeat the full movement.",
        };
      } else if (livePred !== exerciseName) {
        if (result.status === "Correct") {
          result = {
            status: "Needs Improvement",
            similarity: Math.min(Number(result.similarity || 0), 55).toFixed(1),
            message: `Movement looks closer to ${livePred}, not ${exerciseName}. Improve direction and posture.`,
          };
        } else {
          result = {
            status: "Incorrect",
            similarity: Math.min(Number(result.similarity || 0), 40).toFixed(1),
            message: `Incorrect exercise pattern. Model detected ${livePred} while selected exercise is ${exerciseName}.`,
          };
        }
      }
    }

    if (exerciseName === "ABD" && result.status === "Correct") {
      const baseY = Math.max(base.rangeY || 1, 1);
      const currentY = Math.max(repFeatures.rangeY || 0, 0);
      const yRatio = currentY / baseY;

      if (yRatio < 0.6) {
        result = {
          status: "Needs Improvement",
          similarity: Math.min(Number(result.similarity || 0), 58).toFixed(1),
          message:
            "Abduction movement is not wide enough to the side. Lift sideways with better range.",
        };
      }
    }

    repCountRef.current += 1;
    setRepCount(repCountRef.current);

    if (result.status === "Correct") {
      setCorrectReps((prev) => prev + 1);
    } else if (result.status === "Needs Improvement") {
      setNeedsReps((prev) => prev + 1);
    } else {
      setIncorrectReps((prev) => prev + 1);
    }

    setRepHistory((prev) => [
      ...prev.slice(-9),
      {
        rep: repCountRef.current,
        similarity: Number(result.similarity || 0),
        status: result.status,
      },
    ]);

    setMainFeedback(
      `Rep ${repCountRef.current}: ${result.status}. ${result.message} Similarity: ${result.similarity}%`
    );
  }

  function detectPendulumRep(newReading, base) {
    const axis = base.dominantAxis || "angle_x";
    const baseAxis = base.axisFeatures[axis];

    if (!baseAxis) return;

    const value = Number(newReading[axis] || 0);
    const now = Date.now();

    const center = (baseAxis.min + baseAxis.max) / 2;
    const amplitude = Math.max(baseAxis.range / 2, 2);
    const threshold = amplitude * 0.35;

    let side = null;

    if (value > center + threshold) {
      side = "right";
    } else if (value < center - threshold) {
      side = "left";
    }

    if (!side) {
      if (repStateRef.current !== "idle") {
        currentRepSamplesRef.current.push(newReading);
      }
      return;
    }

    if (pendulumLastSideRef.current === null) {
      pendulumLastSideRef.current = side;
      currentRepSamplesRef.current = [newReading];
      repStateRef.current = "swinging";
      return;
    }

    currentRepSamplesRef.current.push(newReading);

    if (currentRepSamplesRef.current.length > 250) {
      currentRepSamplesRef.current = currentRepSamplesRef.current.slice(-250);
    }

    if (side !== pendulumLastSideRef.current) {
      if (now - lastRepTimeRef.current > 700) {
        lastRepTimeRef.current = now;

        const repFeatures = extractMovementFeatures(currentRepSamplesRef.current);

        finishRepEvaluation(repFeatures, "PEN");

        currentRepSamplesRef.current = [newReading];
        pendulumLastSideRef.current = side;
      }
    }
  }

  function startCalibration() {
    if (!backend.sensor_connected && !demoMode) {
      setMainFeedback("Start the WIT sensor before calibration.");
      return;
    }

    stopActualSession();

    calibrationDataRef.current = [];
    setCalibrationData([]);

    setIsCalibrating(true);
    isCalibratingRef.current = true;

    setMainFeedback(
      `Calibration started for ${selectedExercise} - ${EXERCISES[selectedExercise]}. Perform 2–3 slow correct repetitions, then click Save Baseline.`
    );
  }

  function saveCalibration() {
    const features = extractMovementFeatures(calibrationDataRef.current);

    if (!features) {
      setMainFeedback("No calibration data recorded. Try again.");
      return;
    }

    setBaseline(features);
    baselineRef.current = features;

    setIsCalibrating(false);
    isCalibratingRef.current = false;

    setMainFeedback(
      `9-axis baseline saved for ${selectedExercise} - ${
        EXERCISES[selectedExercise]
      }. Dominant angle axis: ${
        features.dominantAxis
      }, main ROM: ${features.dominantRange.toFixed(
        2
      )}°. You can now start actual reps.`
    );
  }

  function startActualSession() {
    if (!baselineRef.current) {
      setMainFeedback("Please calibrate this exercise before starting actual reps.");
      return;
    }

    if (!backend.sensor_connected && !demoMode) {
      setMainFeedback("Start the WIT sensor before beginning actual reps.");
      return;
    }

    setIsActualSession(true);
    isActualSessionRef.current = true;

    repStateRef.current = "down";
    lastRepTimeRef.current = 0;
    currentRepSamplesRef.current = [];
    pendulumLastSideRef.current = null;

    setMainFeedback(
      `Actual session started for ${selectedExercise} - ${EXERCISES[selectedExercise]}. Perform reps slowly and return to the starting position each time.`
    );
  }

  function stopActualSession() {
    setIsActualSession(false);
    isActualSessionRef.current = false;

    currentRepSamplesRef.current = [];
    pendulumLastSideRef.current = null;
  }

  function detectRep(newReading) {
    const base = baselineRef.current;

    if (!base) return;

    const exercise = selectedExerciseRef.current;

    if (exercise === "PEN") {
      detectPendulumRep(newReading, base);
      return;
    }

    const axis = base.dominantAxis || EXERCISE_AXIS[exercise] || "angle_x";

    const value = Number(newReading[axis] || 0);
    const now = Date.now();

    const baseAxis = base.axisFeatures[axis];

    if (!baseAxis) return;

    const minValue = baseAxis.min;
    const range = Math.max(baseAxis.range, 1);

    const progress = (value - minValue) / range;

    let lowThreshold = 0.4;
    let highThreshold = 0.65;

    if (exercise === "IR") {
      lowThreshold = 0.45;
      highThreshold = 0.6;
    }

    const isLow = progress < lowThreshold;
    const isHigh = progress > highThreshold;

    if (repStateRef.current === "down" && isHigh) {
      repStateRef.current = "up";
      currentRepSamplesRef.current = [newReading];
      return;
    }

    if (repStateRef.current === "up") {
      currentRepSamplesRef.current.push(newReading);

      if (currentRepSamplesRef.current.length > 250) {
        currentRepSamplesRef.current = currentRepSamplesRef.current.slice(-250);
      }
    }

    if (repStateRef.current === "up" && isLow) {
      if (now - lastRepTimeRef.current > 900) {
        lastRepTimeRef.current = now;

        const currentRepFeatures = extractMovementFeatures(
          currentRepSamplesRef.current
        );

        finishRepEvaluation(currentRepFeatures, exercise);

        currentRepSamplesRef.current = [];
      }

      repStateRef.current = "down";
    }
  }

  function resetSession() {
    stopActualSession();

    setRepCount(0);
    repCountRef.current = 0;

    setCorrectReps(0);
    setNeedsReps(0);
    setIncorrectReps(0);

    setRepHistory([]);

    repStateRef.current = "down";
    lastRepTimeRef.current = 0;
    currentRepSamplesRef.current = [];
    pendulumLastSideRef.current = null;

    setMainFeedback(
      "Session reset. Keep the saved calibration or recalibrate for another user."
    );
  }

  function resetCalibration() {
    stopActualSession();

    setBaseline(null);
    baselineRef.current = null;

    setCalibrationData([]);
    calibrationDataRef.current = [];

    setMainFeedback(
      "Calibration cleared. Select an exercise and record 2–3 correct repetitions."
    );
  }

  const feedbackClass = useMemo(() => {
    if (mainFeedback.includes("Correct")) return "good";
    if (mainFeedback.includes("Needs")) return "warn";
    if (mainFeedback.includes("Incorrect")) return "bad";
    return "";
  }, [mainFeedback]);

  const predictionClass = useMemo(() => {
    if (prediction.exercise === "WRONG") return "bad";
    if (prediction.exercise === "REST") return "warn";
    if (prediction.confidence >= 70) return "good";
    return "";
  }, [prediction]);

  const chartData = liveData.slice(-70);

  const rangeData = useMemo(() => {
    if (chartData.length === 0) {
      return [
        { name: "Angle X", range: 0 },
        { name: "Angle Y", range: 0 },
        { name: "Angle Z", range: 0 },
      ];
    }

    const getRange = (key) => {
      const values = chartData.map((d) => Number(d[key] || 0));
      return Math.max(...values) - Math.min(...values);
    };

    return [
      { name: "Angle X", range: Number(getRange("angle_x").toFixed(1)) },
      { name: "Angle Y", range: Number(getRange("angle_y").toFixed(1)) },
      { name: "Angle Z", range: Number(getRange("angle_z").toFixed(1)) },
    ];
  }, [chartData]);

  const qualityData = [
    { name: "Correct", value: correctReps },
    { name: "Needs Improvement", value: needsReps },
    { name: "Incorrect", value: incorrectReps },
  ];

  const completedQuality = correctReps + needsReps + incorrectReps;

  const sessionScore =
    completedQuality === 0
      ? 0
      : Math.round((correctReps / completedQuality) * 100);

  const workflowStep = useMemo(() => {
    if (!backend.sensor_connected && !demoMode) return 1;
    if (!baseline) return 2;
    if (!isActualSession) return 3;
    return 4;
  }, [backend.sensor_connected, demoMode, baseline, isActualSession]);

  return (
    <div className="dashboard-layout">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-icon">RF</div>
          <div>
            <h2>Rehab Fit</h2>
            <p>Clinical Monitor</p>
          </div>
        </div>

        <nav className="side-nav">
          <button
            className="side-link active"
            onClick={() => scrollToSection("dashboard")}
          >
            Dashboard
          </button>

          <button
            className="side-link"
            onClick={() => scrollToSection("sensor")}
          >
            Live Sensor
          </button>

          <button
            className="side-link"
            onClick={() => scrollToSection("calibration")}
          >
            Calibration
          </button>

          <button
            className="side-link"
            onClick={() => scrollToSection("session")}
          >
            Exercise Session
          </button>

          <button
            className="side-link"
            onClick={() => scrollToSection("reports")}
          >
            Reports
          </button>
        </nav>

        <div className="side-status">
          <p>Device Status</p>
          <strong>
            {backend.sensor_connected ? "Connected" : "Not Connected"}
          </strong>
          <span>{backend.sensor_status}</span>
        </div>
      </aside>

      <main className="app">
        <header className="topbar">
          <div>
            <p className="eyebrow">Physiotherapy Monitoring System</p>
            <h2>Shoulder Rehabilitation Session</h2>
          </div>

          <div className="topbar-actions">
            <div
              className={
                backend.sensor_connected ? "mini-status on" : "mini-status off"
              }
            >
              {backend.sensor_connected ? "Live Sensor" : "Sensor Offline"}
            </div>

            <div
              className={isActualSession ? "mini-status on" : "mini-status off"}
            >
              {isActualSession ? "Session Running" : "Session Idle"}
            </div>
          </div>
        </header>

        <section id="dashboard" className="hero">
          <div>
            <p className="eyebrow">Smart Physiotherapy FitBand</p>
            <h1>Rehab Fit Clinical Dashboard</h1>
            <p className="subtitle">
              Real WIT IMU streaming through a Python BLE backend, machine
              learning exercise detection, personalized 9-axis calibration, live
              repetition counting, and movement-quality feedback.
            </p>
          </div>

          <div className="hero-card">
            <p>System Status</p>
            <h3>{backend.status}</h3>
            <span>{backend.sensor_status}</span>
          </div>
        </section>

        <section className="workflow-card">
          <div
            className={workflowStep >= 1 ? "workflow-step active" : "workflow-step"}
          >
            <span>1</span>
            <p>Start Sensor</p>
          </div>

          <div
            className={workflowStep >= 2 ? "workflow-step active" : "workflow-step"}
          >
            <span>2</span>
            <p>Calibrate Exercise</p>
          </div>

          <div
            className={workflowStep >= 3 ? "workflow-step active" : "workflow-step"}
          >
            <span>3</span>
            <p>Start Actual Reps</p>
          </div>

          <div
            className={workflowStep >= 4 ? "workflow-step active" : "workflow-step"}
          >
            <span>4</span>
            <p>Track Feedback</p>
          </div>
        </section>

        <section id="sensor" className="grid top-grid">
          <div className="card">
            <p className="eyebrow">Connection</p>
            <h2>Sensor Control</h2>

            <div className="button-row">
              <button onClick={startRealSensor}>Start Real WIT Sensor</button>
              <button className="secondary" onClick={stopRealSensor}>
                Stop Sensor
              </button>
            </div>

            <div className="button-row">
              <button className="demo" onClick={startDemoMode}>
                Start Demo Mode
              </button>
              <button className="secondary" onClick={stopDemoMode}>
                Stop Demo
              </button>
            </div>

            <div
              className={
                backend.sensor_connected ? "pill connected" : "pill disconnected"
              }
            >
              {backend.sensor_connected ? "Connected / Running" : "Not Connected"}
            </div>

            <p className="small-text">
              Model window: <strong>{samples}</strong> / 100 samples
            </p>

            <p className="small-text">
              Close WITMotion before starting the real WIT sensor.
            </p>
          </div>

          <div className={`card prediction-card ${predictionClass}`}>
            <p className="eyebrow">Live Detection</p>
            <h2>Exercise Classification</h2>

            <p className="label">Predicted Exercise</p>
            <h1>{prediction.exercise}</h1>

            <div className="confidence-bar">
              <div
                className="confidence-fill"
                style={{
                  width: `${Math.min(Math.max(prediction.confidence, 0), 100)}%`,
                }}
              ></div>
            </div>

            <p className="confidence">{prediction.confidence}% confidence</p>
            <p className="small-text">{prediction.feedback}</p>
          </div>
        </section>

        <section id="calibration" className="card calibration-card">
          <div className="section-header">
            <div>
              <p className="eyebrow">Personalized Setup</p>
              <h2>Exercise Calibration</h2>
            </div>

            <select
              value={selectedExercise}
              onChange={(e) => {
                setSelectedExercise(e.target.value);
                resetSession();
                resetCalibration();
              }}
            >
              <option value="ABD">ABD - Abduction</option>
              <option value="FEL">FEL - Forward Elevation</option>
              <option value="IR">IR - Internal Rotation</option>
              <option value="PEN">PEN - Pendulum</option>
            </select>
          </div>

          <div className="calibration-layout">
            <div className="step-box">
              <h3>Step 1: Calibrate for the user</h3>
              <p>
                Perform 2–3 correct repetitions. The system stores a 9-axis
                movement fingerprint using acceleration, gyroscope, and angle
                values.
              </p>

              <div className="button-row">
                <button onClick={startCalibration}>Start Calibration</button>

                <button className="secondary" onClick={saveCalibration}>
                  Save Baseline
                </button>

                <button className="secondary" onClick={resetCalibration}>
                  Clear Calibration
                </button>
              </div>
            </div>

            <div className="step-box">
              <h3>Step 2: Start actual repetitions</h3>
              <p>
                After calibration, reps are counted only during the actual
                session. Each completed rep is compared with the saved 9-axis
                baseline and live ML prediction.
              </p>

              <div className="button-row">
                <button onClick={startActualSession}>Start Actual Reps</button>

                <button className="secondary" onClick={stopActualSession}>
                  Stop Actual Reps
                </button>

                <button className="secondary" onClick={resetSession}>
                  Reset Session
                </button>
              </div>
            </div>
          </div>

          <div className={`feedback-box ${feedbackClass}`}>
            <h3>Clinical Feedback</h3>
            <p>{mainFeedback}</p>
          </div>

          <div className="grid two-grid">
            <div className="mini-panel">
              <h3>Session Status</h3>
              <p>
                {isCalibrating
                  ? "🔴 Recording calibration reps"
                  : "Calibration idle"}
              </p>
              <p>
                {isActualSession
                  ? "🟢 Actual session running"
                  : "Actual session not started"}
              </p>
              <p>Calibration samples: {calibrationData.length}</p>
              <p>
                Selected exercise: {selectedExercise} -{" "}
                {EXERCISES[selectedExercise]}
              </p>
            </div>

            <div className="mini-panel">
              <h3>Saved Baseline</h3>

              {baseline ? (
                <>
                  <p>Baseline type: 9-axis movement fingerprint</p>
                  <p>Dominant angle axis: {baseline.dominantAxis}</p>
                  <p>Main ROM: {baseline.dominantRange.toFixed(2)}°</p>
                  <p>Angle X range: {baseline.rangeX.toFixed(2)}°</p>
                  <p>Angle Y range: {baseline.rangeY.toFixed(2)}°</p>
                  <p>Angle Z range: {baseline.rangeZ.toFixed(2)}°</p>
                  <p>Sensor axes used: 9</p>
                </>
              ) : (
                <p>No baseline saved yet.</p>
              )}
            </div>
          </div>
        </section>

        <section id="session" className="grid stats-grid">
          <StatCard title="Total Reps" value={repCount} note="Actual session only" />
          <StatCard title="Correct Reps" value={correctReps} note="Good form" />
          <StatCard
            title="Needs Improvement"
            value={needsReps}
            note="Partial form"
          />
          <StatCard
            title="Session Score"
            value={`${sessionScore}%`}
            note="Quality score"
          />
        </section>

        <section className="grid readings-grid">
          <ReadingCard title="Acceleration X" value={reading.ax} unit="g" />
          <ReadingCard title="Acceleration Y" value={reading.ay} unit="g" />
          <ReadingCard title="Acceleration Z" value={reading.az} unit="g" />
          <ReadingCard title="Gyro X" value={reading.wx} unit="°/s" />
          <ReadingCard title="Gyro Y" value={reading.wy} unit="°/s" />
          <ReadingCard title="Gyro Z" value={reading.wz} unit="°/s" />
          <ReadingCard title="Angle X" value={reading.angle_x} unit="°" />
          <ReadingCard title="Angle Y" value={reading.angle_y} unit="°" />
          <ReadingCard title="Angle Z" value={reading.angle_z} unit="°" />
        </section>

        <section id="reports" className="grid charts-grid">
          <div className="card chart-card large-chart">
            <div className="section-header">
              <div>
                <p className="eyebrow">Live Movement</p>
                <h2>Shoulder Angle Trend</h2>
              </div>
              <p className="small-text">Real-time angle X, Y, and Z</p>
            </div>

            <ResponsiveContainer width="100%" height={285}>
              <LineChart data={chartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8e2ec" />
                <XAxis dataKey="index" />
                <YAxis />
                <Tooltip />
                <Line
                  type="monotone"
                  dataKey="angle_x"
                  name="Angle X"
                  stroke="#1d9bf0"
                  strokeWidth={3}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="angle_y"
                  name="Angle Y"
                  stroke="#14b8a6"
                  strokeWidth={3}
                  dot={false}
                />
                <Line
                  type="monotone"
                  dataKey="angle_z"
                  name="Angle Z"
                  stroke="#f59e0b"
                  strokeWidth={3}
                  dot={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          <div className="card chart-card">
            <p className="eyebrow">Range of Motion</p>
            <h2>Current ROM</h2>

            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={rangeData}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8e2ec" />
                <XAxis dataKey="name" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="range" name="Range °" fill="#1d9bf0" radius={10} />
              </BarChart>
            </ResponsiveContainer>
          </div>

          <div className="card chart-card">
            <p className="eyebrow">Rep Quality</p>
            <h2>Feedback Split</h2>

            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={qualityData}
                  dataKey="value"
                  nameKey="name"
                  outerRadius={90}
                  innerRadius={55}
                  paddingAngle={3}
                >
                  {qualityData.map((_, index) => (
                    <Cell key={index} fill={QUALITY_COLORS[index]} />
                  ))}
                </Pie>
                <Tooltip />
              </PieChart>
            </ResponsiveContainer>
          </div>

          <div className="card chart-card large-chart">
            <p className="eyebrow">Rep Evaluation</p>
            <h2>Similarity History</h2>

            <ResponsiveContainer width="100%" height={260}>
              <AreaChart data={repHistory}>
                <CartesianGrid strokeDasharray="3 3" stroke="#d8e2ec" />
                <XAxis dataKey="rep" />
                <YAxis domain={[0, 100]} />
                <Tooltip />
                <Area
                  type="monotone"
                  dataKey="similarity"
                  name="Similarity %"
                  stroke="#14b8a6"
                  fill="#14b8a6"
                  fillOpacity={0.18}
                  strokeWidth={3}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>
        </section>

        <section className="card">
          <p className="eyebrow">Recent Repetitions</p>
          <h2>Rep Feedback Log</h2>

          {repHistory.length === 0 ? (
            <p className="small-text">
              No actual repetitions evaluated yet. Start actual reps after
              calibration.
            </p>
          ) : (
            <div className="history-list">
              {repHistory.map((rep, index) => (
                <div className="history-item" key={index}>
                  <span>Rep {rep.rep}</span>
                  <strong>{rep.status}</strong>
                  <em>{rep.similarity}% similarity</em>
                </div>
              ))}
            </div>
          )}
        </section>
      </main>
    </div>
  );
}

function ReadingCard({ title, value, unit }) {
  return (
    <div className="reading-card">
      <p>{title}</p>
      <h2>{Number(value || 0).toFixed(2)}</h2>
      <span>{unit}</span>
    </div>
  );
}

function StatCard({ title, value, note }) {
  return (
    <div className="stat-card">
      <p>{title}</p>
      <h2>{value}</h2>
      <span>{note}</span>
    </div>
  );
}

export default App;
