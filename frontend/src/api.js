import axios from 'axios';

const API_URL = ''; // Relative path for production

export const api = axios.create({
  baseURL: API_URL,
});

export const uploadImages = async (files, split, classname) => {
  const formData = new FormData();
  files.forEach(file => {
    formData.append('files', file);
  });
  formData.append('split', split);
  formData.append('classname', classname);

  return api.post('/upload', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};

export const startTraining = async (class1, class2) => {
  return api.post('/train', { class1_name: class1, class2_name: class2 });
};

export const getStatus = async () => {
  return api.get('/status');
};

export const predictImage = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/predict', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};

export const predictFun = async (file) => {
  const formData = new FormData();
  formData.append('file', file);
  return api.post('/predict-fun', formData, {
    headers: { 'Content-Type': 'multipart/form-data' }
  });
};
