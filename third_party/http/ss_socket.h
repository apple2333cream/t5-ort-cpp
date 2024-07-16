#ifndef GRG_SOCKET_H
#define GRG_SOCKET_H
#include <winsock2.h>
#pragma comment(lib, "ws2_32.lib")
int grg_bind(_In_ SOCKET s,
	const struct sockaddr FAR* name,
	_In_ int namelen)
{
	return ::bind(s, name, namelen);
}

#endif 
