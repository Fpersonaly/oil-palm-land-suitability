function doLogin() {

  const email = document.getElementById('email').value;
  const pass = document.getElementById('password').value;
  const err = document.getElementById('error-msg');

  if (
      email === 'peneliti@itsi.ac.id' &&
      pass === 'itsi2025'
  ) {

      window.location.href = 'dashboard.html';

  } else {

      err.style.display = 'block';

  }
}

const passwordField = document.getElementById('password');

if (passwordField) {
  passwordField.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
          doLogin();
      }
  });
}

function doLogout() {
  window.location.href = 'index.html';
}