import React, { useState } from 'react'
import axiosInstance from '../axiosinstance'
import { FontAwesomeIcon } from '@fortawesome/react-fontawesome'
import { faSpinner } from '@fortawesome/free-solid-svg-icons'

const Register = () => {
  const [username, setUsername] = useState('')
  const [email, setEmail] = useState('')
  const [password, setPassword] = useState('')
  const [errors, setErrors] = useState({})
  const [success, setSuccess] = useState(false)
  const [loading, setLoading] = useState(false)

  const handleRegistration = async (e) => {
    e.preventDefault()
    setLoading(true)

    const userData = { username, email, password }

    try {
      const response = await axiosInstance.post('/register/', userData)

      console.log('response.data ==>', response.data)
      setErrors({})
      setSuccess(true)
    } catch (error) {
      if (error.response) {
        setErrors(error.response.data)
        console.error('Registration error:', error.response.data)
      } else {
        console.error('Network error:', error)
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className='container'>
      <div className='row justify-content-center'>
        <div className='col-md-6 bg-light-dark p-5 rounded'>
          <h3 className='text-light text-center mb-4'>Create an Account</h3>

          <form onSubmit={handleRegistration}>
            <div className='mb-3'>
              <input
                type='text'
                className='form-control'
                placeholder='Username'
                value={username}
                onChange={(e) => setUsername(e.target.value)}
              />
              {errors.username && (
                <small className='text-danger'>{errors.username}</small>
              )}
            </div>

            <div className='mb-3'>
              <input
                type='email'
                className='form-control'
                placeholder='Email address'
                value={email}
                onChange={(e) => setEmail(e.target.value)}
              />
            </div>

            <div className='mb-3'>
              <input
                type='password'
                className='form-control'
                placeholder='Set password'
                value={password}
                onChange={(e) => setPassword(e.target.value)}
              />
              {errors.password && (
                <small className='text-danger'>{errors.password}</small>
              )}
            </div>

            {success && (
              <div className='alert alert-success'>
                Registration Successful
              </div>
            )}

            <button
              type='submit'
              className='btn btn-info d-block mx-auto'
              disabled={loading}
            >
              {loading ? (
                <>
                  <FontAwesomeIcon icon={faSpinner} spin /> Please wait...
                </>
              ) : (
                'Register'
              )}
            </button>
          </form>
        </div>
      </div>
    </div>
  )
}

export default Register
