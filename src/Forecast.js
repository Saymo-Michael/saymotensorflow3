import React, { useState } from 'react';
import Papa from 'papaparse';
import './Forecast.css';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import * as tf from '@tensorflow/tfjs';
import { Table, Form, Button, Container, Row, Col } from 'react-bootstrap';

const Forecast = () => {
  const [data, setData] = useState([]);
  const [predictions, setPredictions] = useState([]);
  const [fileError, setFileError] = useState('');
  const [loading, setLoading] = useState(false);
  const [maxStudents, setMaxStudents] = useState(30); // Default maximum students per section

  // Handle CSV file upload
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
        header: true,
        skipEmptyLines: true,
        complete: (result) => {
          const processedData = preprocessData(result.data);
          if (processedData.length === 0) {
            setFileError('No valid data found in the CSV file.');
          } else {
            setData(processedData);
            console.log('Processed Data:', processedData); // Debugging
          }
        },
        error: (error) => {
          console.error('Error parsing CSV:', error);
          setFileError('Failed to parse the CSV file. Please check the format.');
        },
      });
    } else {
      setFileError('No file selected.');
    }
  };

  // Data preprocessing function
  const preprocessData = (rawData) => {
    const requiredFields = ['semester', 'course code', 'total number of students enrolled'];

    return rawData
      .filter((row) => {
        // Ensure all required fields are present
        return requiredFields.every((field) => row[field]);
      })
      .map((entry) => {
        return {
          semester: entry['semester'] || entry['SEMESTER'] || '',
          courseCode: entry['course code'] || entry['COURSE CODE'] || '',
          totalStudents: parseInt(
            entry['total number of students enrolled'] ||
              entry['TOTAL NUMBER OF STUDENTS ENROLLED'] ||
              '0',
            10
          ),
        };
      })
      .filter((entry) => entry.totalStudents > 0); // Filter out invalid rows
  };

  // Train and predict using TensorFlow.js
  const trainAndPredict = async () => {
    if (!data || data.length === 0) {
      setFileError('Please upload a CSV file first.');
      return;
    }

    setLoading(true);

    try {
      const xs = tf.tensor2d(data.map((entry, index) => [index])); // X-axis (indexes)
      const ys = tf.tensor1d(data.map((entry) => entry.totalStudents)); // Y-axis (total students)

      const model = tf.sequential();
      model.add(tf.layers.dense({ units: 64, inputShape: [1], activation: 'relu' }));
      model.add(tf.layers.dense({ units: 1 }));

      model.compile({ optimizer: tf.train.adam(0.01), loss: 'meanSquaredError' });

      await model.fit(xs, ys, { epochs: 100 });

      const predictedEnrollments = data.map((entry, index) => {
        const inputTensor = tf.tensor2d([[index]]);
        const predictedValue = model.predict(inputTensor).dataSync()[0];
        const sectionsNeeded = Math.ceil(predictedValue / maxStudents);
        return {
          ...entry,
          predictedEnrollment: Math.round(predictedValue),
          sectionsNeeded,
        };
      });

      setPredictions(predictedEnrollments);
      console.log('Predictions:', predictedEnrollments); // Debugging
    } catch (error) {
      console.error('Error during model training or prediction:', error);
      setFileError('An error occurred during prediction.');
    }

    setLoading(false);
  };

  return (
    <Container className="forecast-container">
      <h1>Enrollment Forecast</h1>

      <Row className="mb-3">
        <Col>
          <Form.Group>
            <Form.Label>Upload CSV File</Form.Label>
            <Form.Control type="file" accept=".csv" onChange={handleFileUpload} />
            {fileError && <p className="text-danger mt-2">{fileError}</p>}
          </Form.Group>
        </Col>
      </Row>

      {/* <Row className="mb-3">
        <Col>
          <Form.Group>
            <Form.Label>Maximum Students Per Section</Form.Label>
            <Form.Control
              type="number"
              value={maxStudents}
              onChange={(e) => setMaxStudents(parseInt(e.target.value, 10) || 30)}
            />
          </Form.Group>
        </Col>
      </Row> */}

      <Button onClick={trainAndPredict} disabled={loading}>
        {loading ? 'Predicting...' : 'Predict Enrollment'}
      </Button>

      {predictions.length > 0 && (
        <>
          <h2 className="mt-4">Predicted Enrollments</h2>
          <Table striped bordered hover responsive>
            <thead>
              <tr>
                <th>Semester</th>
                <th>Course Code</th>
                <th>Total Students Enrolled</th>
                <th>Predicted Enrollment</th>
                <th>Sections Needed</th>
              </tr>
            </thead>
            <tbody>
              {predictions.map((entry, index) => (
                <tr key={index}>
                  <td>{entry.semester}</td>
                  <td>{entry.courseCode}</td>
                  <td>{entry.totalStudents}</td>
                  <td>{entry.predictedEnrollment}</td>
                  <td>{entry.sectionsNeeded}</td>
                </tr>
              ))}
            </tbody>
          </Table>

          <h2 className="mt-4">Enrollment Forecast Chart</h2>
          <ResponsiveContainer width="100%" height={400}>
            <LineChart data={predictions} margin={{ top: 20, right: 30, left: 20, bottom: 60 }}>
              <CartesianGrid strokeDasharray="3 3" />
              <XAxis
                dataKey="courseCode"
                label={{ value: 'Course Code', position: 'bottom', offset: 20 }}
              />
              <YAxis
                label={{ value: 'Predicted Enrollment', angle: -90, position: 'insideLeft' }}
              />
              <Tooltip />
              <Legend wrapperStyle={{ paddingTop: 30 }} />
              <Line type="monotone" dataKey="predictedEnrollment" stroke="#82ca9d" />
            </LineChart>
          </ResponsiveContainer>
        </>
      )}
    </Container>
  );
};

export default Forecast;
