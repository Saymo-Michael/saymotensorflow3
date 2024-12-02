import React, { useState } from 'react';
import Papa from 'papaparse';
import * as tf from '@tensorflow/tfjs';
import { Container, Form, Button, Table } from 'react-bootstrap';
import { Chart as ChartJS, BarElement, CategoryScale, LinearScale } from 'chart.js';
import { Bar } from 'react-chartjs-2';

ChartJS.register(CategoryScale, LinearScale, BarElement);

const Forecast = () => {
  const [data, setData] = useState([]);
  const [semester, setSemester] = useState('');
  const [courseCode, setCourseCode] = useState('');
  const [maxStudents, setMaxStudents] = useState(30);
  const [predictions, setPredictions] = useState([]);
  const [loading, setLoading] = useState(false);
  const [fileError, setFileError] = useState('');

  // Handle CSV upload
  const handleFileUpload = (event) => {
    const file = event.target.files[0];
    if (file) {
      if (file.type !== 'text/csv') {
        setFileError('Invalid file type. Please upload a CSV file.');
        return;
      } else {
        setFileError('');
      }

      Papa.parse(file, {
        complete: (result) => {
          const processedData = preprocessData(result.data);
          if (processedData.length === 0) {
            setFileError('No valid data found in the CSV file.');
          } else {
            setData(processedData);
          }
        },
        header: true,
      });
    } else {
      setFileError('No file selected.');
    }
  };

  // Preprocess data
  const preprocessData = (rawData) => {
    return rawData
      .map((row) => ({
        courseCode: row.course_code,
        enrollment: parseInt(row.total_enrolled || 0, 10),
      }))
      .filter((row) => row.courseCode && !isNaN(row.enrollment));
  };

  // Train and predict enrollment
  const predictEnrollment = async () => {
    if (!data.length || !semester || !courseCode) {
      setFileError('Please upload data and fill all input fields.');
      return;
    }

    setLoading(true);
    try {
      const xs = tf.tensor2d(data.map((d, i) => [i])); // Use row index as input
      const ys = tf.tensor1d(data.map((d) => d.enrollment)); // Enrollment as output

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 10, inputShape: [1], activation: 'relu' }));
      model.add(tf.layers.dense({ units: 1 }));
      model.compile({ optimizer: 'adam', loss: 'meanSquaredError' });

      await model.fit(xs, ys, { epochs: 20 });

      const predictedEnrollment = model.predict(tf.tensor2d([[data.length]])).dataSync()[0];
      const sectionsNeeded = Math.ceil(predictedEnrollment / maxStudents);

      setPredictions((prev) => [
        ...prev,
        { semester, courseCode, predictedEnrollment, sectionsNeeded },
      ]);
    } catch (error) {
      setFileError('Error during prediction: ' + error.message);
    }
    setLoading(false);
  };

  // Chart data
  const chartData = {
    labels: predictions.map((p) => p.courseCode),
    datasets: [
      {
        label: 'Predicted Enrollment',
        data: predictions.map((p) => p.predictedEnrollment),
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        borderColor: 'rgba(75, 192, 192, 1)',
        borderWidth: 1,
      },
    ],
  };

  return (
    <Container>
      <h1 className="text-center my-4">Enrollment Forecast</h1>
      <Form>
        <Form.Group className="mb-3">
          <Form.Label>Semester</Form.Label>
          <Form.Control
            type="text"
            placeholder="e.g., 2022-1"
            value={semester}
            onChange={(e) => setSemester(e.target.value)}
          />
        </Form.Group>
        <Form.Group className="mb-3">
          <Form.Label>Course Code</Form.Label>
          <Form.Control
            type="text"
            placeholder="e.g., ITE102"
            value={courseCode}
            onChange={(e) => setCourseCode(e.target.value)}
          />
        </Form.Group>
        <Form.Group className="mb-3">
          <Form.Label>Total Students Enrolled (Max per Section)</Form.Label>
          <Form.Control
            type="number"
            value={maxStudents}
            onChange={(e) => setMaxStudents(Number(e.target.value))}
            min="1"
          />
        </Form.Group>
        <Form.Group className="mb-3">
          <Form.Label>Upload CSV File</Form.Label>
          <Form.Control type="file" accept=".csv" onChange={handleFileUpload} />
          {fileError && <p className="text-danger">{fileError}</p>}
        </Form.Group>
        <Button variant="primary" onClick={predictEnrollment} disabled={loading}>
          {loading ? 'Predicting...' : 'Predict Enrollment'}
        </Button>
      </Form>
      {predictions.length > 0 && (
        <>
          <h2 className="text-center my-4">Predicted Results</h2>
          <Table striped bordered hover responsive>
            <thead>
              <tr>
                <th>Semester</th>
                <th>Course Code</th>
                <th>Predicted Enrollment</th>
                <th>Sections Needed</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((p, index) => (
                <tr key={index}>
                  <td>{p.semester}</td>
                  <td>{p.courseCode}</td>
                  <td>{Math.round(p.predictedEnrollment)}</td>
                  <td>{p.sectionsNeeded}</td>
                </tr>
              ))}
            </tbody>
          </Table>
          <div className="mt-4">
            <Bar data={chartData} options={{ responsive: true, plugins: { legend: { position: 'top' } } }} />
          </div>
        </>
      )}
    </Container>
  );
};

export default Forecast;
