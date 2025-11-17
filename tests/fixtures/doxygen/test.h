#ifndef TEST_H
#define TEST_H

/**
 * \brief Gets a device handle
 * \param device Pointer to store device ID
 * \param pciBusId The PCI bus ID string
 * \return Status code
 * \note This function may fail
 * \notefnerr This is important
 */
int get_device(int *device, const char *pciBusId);

/**
 * Sets the \p alpha parameter to \c value
 * \param alpha The alpha value
 * \param value The \em new value to set
 * \return Success status
 * \sa get_device
 */
int set_alpha(float *alpha, float value);

/**
 * \brief Cleanup function
 * \p handle is the \b critical resource
 */
void cleanup(void *handle);

#endif
