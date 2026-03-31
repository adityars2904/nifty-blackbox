import { GoogleLogin, googleLogout } from '@react-oauth/google';
import { jwtDecode } from 'jwt-decode';

export default function GoogleAuth({ onLogin, onLogout, user }) {
  const handleSuccess = (credentialResponse) => {
    const decoded = jwtDecode(credentialResponse.credential);
    onLogin({
      name: decoded.name,
      email: decoded.email,
      picture: decoded.picture,
      token: credentialResponse.credential,
    });
  };

  if (user) {
    return (
      <div className="flex items-center gap-3">
        <div className="w-8 h-8 rounded-full border border-border overflow-hidden flex items-center justify-center bg-highlight/30 text-sm font-bold flex-shrink-0">
          {user.picture ? (
            <img
              src={user.picture}
              alt={user.name || user.email}
              className="w-full h-full object-cover rounded-full"
              onError={(e) => {
                e.target.style.display = 'none';
                e.target.parentElement.textContent = user.name?.[0]?.toUpperCase() || user.email[0].toUpperCase();
              }}
            />
          ) : (
            <span>{user.name?.[0]?.toUpperCase() || user.email[0].toUpperCase()}</span>
          )}
        </div>
        <span className="text-sm text-text-secondary hidden sm:inline">{user.email}</span>
        <button
          onClick={() => {
            googleLogout();
            onLogout();
          }}
          className="px-3 py-1.5 rounded text-xs font-medium text-text-secondary border border-border
                     hover:text-danger hover:border-danger/30 hover:bg-danger/10 transition-all"
        >
          Sign Out
        </button>
      </div>
    );
  }

  return (
    <GoogleLogin
      onSuccess={handleSuccess}
      onError={() => alert('Login Failed')}
      theme="filled_black"
      shape="pill"
      size="medium"
    />
  );
}
